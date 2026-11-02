#!/usr/bin/env python3
"""DeFT-RAG: decomposing a multi-hop question into self-contained sub-questions, one at a time.

The LLM performs three fine-grained tasks: write the next self-contained sub-question from the original question and
the QA pairs found so far (LLM_sub), answer a question from its selected passages (LLM_ans), and answer the original
question from the QA table (LLM_fin). A lightweight decision model (DM) makes the yes/no decisions (passage relevance,
direct answer, enough facts, and -- with the QA cache -- same question); the QA table is updated by rules.

    gate     the DM judges the question's top-10 passages: if they are sufficient (>= GATE_SUFF_HYBRID), the estimated
             depth is <= DEPTH_TAU and the question is not a composite family relation (KIN_Q), the question is
             answered directly from the selected passages; otherwise it goes to the loop.
    loop     at most MAX_STEPS sub-questions. A sub-question that repeats one already asked gets ONE regeneration
             with a note; a second repeat ends the loop ("stuck"). Each sub-question is answered from the DM-selected
             top-10; an empty answer gets ONE retry with the selected top-20. Still empty -> the sub-question goes to
             `failed` and the next prompt asks for something different. Otherwise (sub-question, answer) joins the QA
             table, and the DM's enough-facts decision (>= GAP_TAU) ends the loop ("gap-closed").
    final    enough facts: LLM_fin answers from the QA table (ONE retry if empty). Otherwise (cap, stuck, no QA pairs):
             answer the question from its own selected top-20 passages, with ONE retry with the QA pairs added.
Every stage allows at most one retry, and every retry changes its input (temperature 0).

QA cache (cache.py, off unless DEFTRAG_REUSE=1): an exact (normalised) match from another question is reused as-is;
otherwise the 5 nearest answered entries by retrieval embedding go to one strict same-question decision, and the best
is reused if its probability >= SAME_TAU.

The prompt strings below are exactly those used for the results in the paper; do not paraphrase them.
"""
import os, sys, json, time, re, urllib.request, urllib.error

VERSION = "deftrag"
# Rules on top of the base loop, as switches (DEFTRAG_FIXES; unset = the accepted set used in the paper):
#   hybridgate   answer directly only if sufficient >= GATE_SUFF_HYBRID, depth <= DEPTH_TAU and the question is not
#                a composite family relation (KIN_Q); everything else goes to the loop
#   stated       sub-questions ask for facts a document states, not calculated values (dates, not ages)
#   samekind     sub-questions ask for the kind of fact the question names (nationality, not birthplace)
#   fbfacts      a non-answer from the passages fallback gets ONE retry with the QA pairs found + passages
#   dates        the final answer compares dates year first; the person born earlier is older
#   samecountry  "same country / nationality" questions only (keyword gate SAMECOUNTRY_Q)
#   reason       lifespan questions only (keyword gate LIFESPAN_Q): the final answer is worked out in 1-2 lines
#   notapplicable "Not applicable" counts as a non-answer
# Not in the accepted set (ablations only): nogate (no pre-loop gate), exnote, resolvefirst, norelfilter (no passage
# selection: the reader gets the plain ranked top-5 / top-10 lists).
ACCEPTED_FIXES = ("hybridgate", "stated", "samekind", "fbfacts", "dates", "samecountry", "reason", "notapplicable")
ALL_FIXES = ACCEPTED_FIXES + ("nogate", "exnote", "resolvefirst", "norelfilter")
FIXES = frozenset(x for x in os.environ.get("DEFTRAG_FIXES", ",".join(ACCEPTED_FIXES)).split(",") if x)
if FIXES - set(ALL_FIXES): raise SystemExit(f"unknown DEFTRAG_FIXES {sorted(FIXES - set(ALL_FIXES))}")
# Sub-query reuse cache: OFF unless DEFTRAG_REUSE=1, which also needs DEFTRAG_CACHE_DIR.
REUSE = os.environ.get("DEFTRAG_REUSE", "0") == "1"
if REUSE and not os.environ.get("DEFTRAG_CACHE_DIR"):
    raise SystemExit("DEFTRAG_REUSE=1 needs DEFTRAG_CACHE_DIR (where the reuse cache is stored)")
import torch

# The LogicRAG release provides the benchmark corpora (its dataset/ folder); retrieval is deftrag/retriever.py.
LG = os.path.abspath(os.path.expanduser(os.environ.get("LOGICRAG_DIR", "LogicRAG")))

# The decision model (DM): the fine-tuned StartLux-Decision-4B, served by the StartLux-Decision release's server
# (POST /v1/systemone). DEFTRAG_DECIDER_KEY is sent as a bearer token if set (a hosted DM); a local server needs none.
DECIDER_URL = os.environ.get("DEFTRAG_DECIDER_URL", "http://localhost:8090/v1/systemone")
KEY = os.environ.get("DEFTRAG_DECIDER_KEY", "")
# The backbone LLM: any OpenAI-compatible chat-completions endpoint. Defaults: a local Ollama server with Qwen3.6.
# For gpt-4o-mini set DEFTRAG_LLM=gpt-4o-mini, DEFTRAG_LLM_URL=<your endpoint>/chat/completions, DEFTRAG_LLM_KEY=<key>.
LLM_MODEL = os.environ.get("DEFTRAG_LLM", "qwen3.6:35b-a3b-q8_0")
LLM_URL = os.environ.get("DEFTRAG_LLM_URL", "http://localhost:11434/v1/chat/completions")
LLM_KEY = os.environ.get("DEFTRAG_LLM_KEY", "ollama")
DM_MODEL = os.environ.get("DEFTRAG_DECIDER_MODEL", "deftrag-dm")    # the name the DM server was started with (--name)
# Request format of the DM. "v2" (default; all results in the paper): the format the fine-tuned DM is trained for --
# relevance asked yes/no (_rel_yn, kept if p(yes) >= REL_TAU_YN) and depth asked from the question alone, in its own
# request (no passages). "v1": the requests of the hosted decision model compared in the paper's overhead appendix
# (relevance as a 4-level score, kept if >= REL_TAU).
DECIDER_FORMAT = os.environ.get("DEFTRAG_DECIDER_FORMAT", "v2")
if DECIDER_FORMAT not in ("v1", "v2"): raise SystemExit(f"DEFTRAG_DECIDER_FORMAT={DECIDER_FORMAT!r}; expected v1 or v2")
# yes/no relevance cut-off; DEFTRAG_REL_TAU_YN sets a value fitted on training-split dev data (decision_model/fit_rel_threshold.py)
REL_TAU_YN = float(os.environ.get("DEFTRAG_REL_TAU_YN", "0.5"))

MAX_STEPS = 6
WIDE_K = 20        # retrieval depth for the ONE retry of an empty answer, and for the final passages
DEPTH_TAU = 2.0
GATE_SUFF = 0.5          # v3.0 gate: answer directly if sufficient >= this and depth <= DEPTH_TAU
GATE_SUFF_HYBRID = float(os.environ.get("DEFTRAG_GATE_SUFF", "0.7"))   # "hybridgate": stricter sufficiency, and composite
# family relations go to the loop. DEFTRAG_GATE_SUFF only for the threshold-sensitivity runs (paper, Appendix B).
# v3.8: 0.7 (was 0.8 in v3.5-v3.7). With 0.8, HotpotQA questions with sufficient 0.7-0.8 went to the loop and did
# worse there (v3.7 HotpotQA 0.858 vs v3.0 0.866); estimated from stored answers, 0.7 gives HotpotQA 0.866,
# 2Wiki 0.878 (unchanged), MuSiQue 0.642.
# composite family relations: answered directly the reader stops one link short (2Wiki kinship questions
# 0.56 direct vs 0.81 through the loop); they are rare elsewhere (MuSiQue 10, HotpotQA 9 of 1000)
KIN_Q = re.compile(r"\bgrand(father|mother|son|daughter|parent|child)|\b(father|mother|son|daughter|brother|sister)[- ]in[- ]law\b|"
                   r"\bstep(father|mother|son|daughter|brother|sister)\b|\b(uncle|aunt|nephew|niece|cousin)\b|\bsibling", re.I)
REL_TAU = 1.0
REL_FLOOR = 2
WEAK_FLOOR = 5     # passages kept when NONE scores >= REL_TAU (weak evidence)
TOP_K = 10
GAP_TAU = 0.5
SAME_TAU = 0.8     # similarity reuse only if the DM judges the two sub-queries the same with p >= this
CACHE_K = 5        # nearest cached sub-queries sent to the SAME judgment
CAP_PASSAGE = 3000
# "norelfilter": no relevance filter, so the reader gets the plain ranked list at the baselines' size: top-5 for every
# question it answers (sub-questions, and the main question's direct answer, whose `sufficient` check then also sees
# those 5); top-10 for the one retry after an empty answer and for the final passages fallback.
if "norelfilter" in FIXES:
    TOP_K, WIDE_K = 5, 10
# Which benchmark's corpus to retrieve from. Set by run_eval.py from its mode; default MuSiQue.
DATASETS = {"musique": "musique_corpus.json", "2wikimultihopqa": "2wikimultihopqa_corpus.json",
            "hotpotqa": "hotpotqa_corpus.json"}
DATASET = os.environ.get("DEFTRAG_DATASET", "musique")
if DATASET not in DATASETS: raise SystemExit(f"DEFTRAG_DATASET={DATASET!r}; expected one of {sorted(DATASETS)}")
CORPUS = DATASETS[DATASET]
# A different corpus file in LogicRAG/dataset/ (training-split runs that collect decision-model training data; the
# test benchmarks never set it).
CUSTOM_CORPUS = os.environ.get("DEFTRAG_CORPUS")
if CUSTOM_CORPUS: CORPUS = CUSTOM_CORPUS

# ---------------- retrieval ----------------
_M = None
def _embedder():
    global _M
    if _M is None:
        from retriever import Retriever
        _M = Retriever(os.path.join(LG, "dataset", CORPUS))
    return _M

def ptext(ci):
    d = _embedder().corpus_list[ci]
    return f"Title: {d['title']}. Content: {d['text']}"[:CAP_PASSAGE]

def embed_query(query):
    return _embedder().encode_query(query)

def retrieve_ids(query, k=TOP_K, emb=None):
    m = _embedder()
    q = embed_query(query) if emb is None else emb
    sims = torch.nn.functional.cosine_similarity(q.unsqueeze(0), m.corpus_embeddings)
    return sims.topk(k).indices.tolist()

def retrieve(query, k=TOP_K):
    return [ptext(ci) for ci in retrieve_ids(query, k)]

def texts(ids):
    return [ptext(ci) for ci in ids]

# ---------------- decision model ----------------
def dm(payload):
    t0 = time.time()
    try:
        return _dm(payload)
    finally:
        COUNTERS["dm_secs"] += time.time() - t0

def _backoff(key, secs):
    """Sleep between retries, counting them (overhead analysis: retries inflate latency)."""
    COUNTERS[key + "_retries"] += 1; COUNTERS[key + "_wait_secs"] += secs
    time.sleep(secs)

def _dm(payload):
    COUNTERS["dm_calls"] += 1
    data = json.dumps(payload).encode()
    last = None
    for att in range(6):
        try:
            req = urllib.request.Request(DECIDER_URL, data=data,
                headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {KEY}"} if KEY else {})})
            return json.load(urllib.request.urlopen(req, timeout=90))["answers"]
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) or 500 <= e.code <= 599:     # any 5xx, incl. 520 seen from the gateway
                last = f"HTTP {e.code}"
                _backoff("dm", min(30, 1.5 * 2 ** att)); continue
            raise RuntimeError(f"decision model HTTP {e.code}") from None
        except Exception as e:
            if att == 5: raise
            last = type(e).__name__
            _backoff("dm", 1.5 * 2 ** att)
    raise RuntimeError(f"decision model: no response after 6 attempts (last error: {last})")

SUFFICIENT = {"type": "noul",
    "instructions": "Do the passages `p1` through `p10` together contain all the facts needed to answer `question` -- whether stated outright or derivable by combining facts across them?",
    "criteria": {"true": "Every fact needed is present across the passages; the answer is stated or derivable by connecting them.",
                 "false": "At least one fact needed to answer is missing and would have to be found elsewhere."}}

DEPTH = {"type": "score",
    "instructions": "How many chained lookups does answering `question` require -- how many intermediate entities or facts must be found in sequence before reaching the final answer? Judge this from the structure of the question itself, regardless of any passages.",
    "criteria": ["None: the answer is a single fact about an entity the question names directly.",
                 "One: find one intermediate entity or fact, then use it to reach the answer.",
                 "Two: two intermediate lookups must be chained before the answer.",
                 "Three or more: a long chain of three or more intermediate lookups."]}

def _rel_q(i):
    return {"type": "score",
            "instructions": f"How essential is `p{i}` for answering `question`?",
            "criteria": ["Irrelevant to the question.",
                         "Loosely related but not needed to answer.",
                         "Helpful: supports the answer but not strictly required.",
                         "Essential: states part of the answer, or a linking fact the reasoning must pass through (e.g., identifying an entity the question refers to only by description)."]}

def _rel_yn(i):
    """Yes/no relevance (open decision model, e.g. fine-tuned StartLux-4B): the dataset gold labels are binary
    (supporting paragraph or not), so the question is binary. Wording = the 'Essential' level of _rel_q.
    The fine-tuning data (decision_model/build_ft_data.py) imports this exact definition."""
    return {"type": "noul",
            "instructions": f"Is `p{i}` needed to answer `question`: does it state part of the answer, or a linking fact the reasoning must pass through (e.g., identifying an entity the question refers to only by description)?",
            "criteria": {"true": f"`p{i}` states part of the answer, or a linking fact needed to reach it.",
                         "false": f"`p{i}` is not needed to answer `question`: it is unrelated or only loosely related."}}

def _dm_scores(query, passages, extra):
    state = {"question": query, **{f"p{i+1}": passages[i] for i in range(len(passages))}}
    qs = dict(extra)
    v2 = DECIDER_FORMAT == "v2"
    if "norelfilter" in FIXES:          # no relevance rows: only the other questions (e.g. sufficient) are asked
        a = dm({"model": DM_MODEL, "state": state, "questions": qs}) if qs else {}
        return a, [1.0] * len(passages)
    qs.update({f"rel{i}": (_rel_yn(i) if v2 else _rel_q(i)) for i in range(1, len(passages) + 1)})
    a = dm({"model": DM_MODEL, "state": state, "questions": qs})
    return a, [a[f"rel{i}"]["noul" if v2 else "score"] for i in range(1, len(passages) + 1)]

def trigger(query, passages):
    """sufficient + depth + rel1..rel10 in ONE call. Pre-loop only: decides
    whether the question can be answered directly. Returns (suff, depth, rels)."""
    if DECIDER_FORMAT == "v2":       # depth from the question alone, in its own request; sufficient + relevance together
        a, rels = _dm_scores(query, passages, {"sufficient": SUFFICIENT})
        d = dm({"model": DM_MODEL, "state": {"question": query}, "questions": {"depth": DEPTH}})
        return a["sufficient"]["noul"], d["depth"]["score"], rels
    a, rels = _dm_scores(query, passages, {"sufficient": SUFFICIENT, "depth": DEPTH})
    return a["sufficient"]["noul"], a["depth"]["score"], rels

def relevance(query, passages):
    """rel1..relK only (K = 10 or 20). Used for sub-queries and the final passages: there is no
    gate there, so `sufficient`/`depth` would only add cost. Returns rels."""
    return _dm_scores(query, passages, {})[1]

def filter_passages(passages, rels):
    """Keep passages scoring >= REL_TAU. If exactly one qualifies, keep the top REL_FLOOR.
    If NONE qualifies, keep the top WEAK_FLOOR: on 124 weak-evidence steps, the best gold
    passage sat at DM score-rank 3-5 in 22 of them, which a floor of 2 threw away."""
    if "norelfilter" in FIXES:
        return list(passages)           # retrieval order, nothing dropped
    tau = REL_TAU_YN if DECIDER_FORMAT == "v2" else REL_TAU
    kept = [i for i, s in enumerate(rels) if s >= tau]
    by_score = sorted(range(len(rels)), key=lambda i: -rels[i])
    if not kept:
        kept = by_score[:WEAK_FLOOR]
    elif len(kept) < REL_FLOOR:
        kept = by_score[:REL_FLOOR]
    return [passages[i] for i in sorted(kept, key=lambda i: -rels[i])]

# ---------------- LLM ----------------
# Ollama degrades silently: a length-truncated or empty reply raises nothing. Count both.
# Overhead analysis: time (secs), tokens as the API reports them (usage), and retries with the time
# spent waiting between them, for the LLM and for the DM.
COUNTERS = {"llm_calls": 0, "llm_length": 0, "llm_empty": 0, "llm_filtered": 0, "dm_calls": 0,
            "cache_exact": 0, "cache_similar": 0, "cache_same_calls": 0,
            "llm_secs": 0.0, "llm_prompt_tokens": 0, "llm_completion_tokens": 0,
            "llm_retries": 0, "llm_wait_secs": 0.0, "dm_secs": 0.0, "dm_retries": 0, "dm_wait_secs": 0.0}

def _llm(prompt, max_tokens=256):
    t0 = time.time()
    try:
        return _llm_call(prompt, max_tokens)
    finally:
        COUNTERS["llm_secs"] += time.time() - t0

def _llm_call(prompt, max_tokens):
    body = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens}
    # reasoning_effort "none" turns off Qwen's thinking mode; Azure rejects the argument (HTTP 400)
    if LLM_MODEL.startswith("qwen"): body["reasoning_effort"] = "none"
    req = urllib.request.Request(LLM_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_KEY}"})
    for attempt in range(8):
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=600))
            ch = resp["choices"][0]
            u = resp.get("usage") or {}
            COUNTERS["llm_prompt_tokens"] += u.get("prompt_tokens", 0)
            COUNTERS["llm_completion_tokens"] += u.get("completion_tokens", 0)
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            # Azure's content filter blocks some prompts (400 content_filter); retrying cannot help,
            # so count it and return no answer, which the loop already handles as a non-answer
            if e.code == 400 and "content_filter" in msg:
                COUNTERS["llm_calls"] += 1; COUNTERS["llm_filtered"] += 1; COUNTERS["llm_empty"] += 1
                return ""
            if e.code not in (408, 429, 500, 502, 503, 504) or attempt == 7: raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 7: raise
        _backoff("llm", min(60, 2 ** attempt))
    out = (ch["message"].get("content") or "").strip()
    COUNTERS["llm_calls"] += 1
    if ch.get("finish_reason") == "length": COUNTERS["llm_length"] += 1
    if ch.get("finish_reason") == "content_filter": COUNTERS["llm_filtered"] += 1
    if not out: COUNTERS["llm_empty"] += 1
    return out

def _ctx(P):
    return "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(P))

ANSWER_T = """Answer the question using ONLY the passages below. Give a short, direct answer -- a few words, not a sentence.

Passages:
{ctx}

Question: {q}
Answer:"""

# fix "fbfacts": ONE retry when the passages fallback returns a non-answer and the loop found facts
# (stuck loops often hold part of the chain). Used only after a non-answer.
FALLBACK_FACTS_T = """Answer the question using ONLY the facts and passages below. Give a short, direct answer -- a few words, not a sentence.

An earlier attempt gave no answer. This is the final answer to the question, so do not reply that the answer is unknown or not stated: combine the facts and passages to give your best answer.

Facts:
{facts}

Passages:
{ctx}

Question: {q}
Answer:"""

# ---- sub-query prompts: one per (facts?, failed?) case ----
_DESC = 'A question may refer to something by description instead of by name, such as "the author of Y". Some descriptions are built on others: "the birthplace of the author of Y" needs "the author of Y" first.'

# Sub-queries must request facts a document states, not derived values (2Wiki: "How old is X?"
# for "which director is older" -> the reader computed ages at death and the comparison flipped;
# 35 loop questions @ 0.686 vs 0.949 when birth dates were asked).
_STATED = """Ask for facts that a document states directly, not for values that must be calculated from them. For example, to find out who is older, ask for each person's date of birth, not how old they are."""

# fix "samekind": 2Wiki "Which country is the composer ... from?" -> "Where was he born?" -> Lviv ->
# Ukraine (gold: Austrian); "which country ... from" questions: 23 of 70 wrong.
_SAMEKIND = """Ask for the same kind of fact the question asks about. For example, if it asks which country someone is from, ask for their nationality, not their place of birth."""
# Placement of the rules (development studies on rebuilt prompt states):
#   v3.3  two paragraphs between "write names in place of descriptions" and its example -> refusals
#         ("I cannot answer ... the facts were not provided"): 26 of 29 rebuilt refusal prompts
#   v3.4/3.5  one merged paragraph at the very END -> no refusals, but over-applied: 27 of 34 steps asked
#         for a birth/release date the question did not need
#   v3.6  one merged paragraph right AFTER the example, before the real question: 0/29 refusals, 35/35 age
#         steps ask a date of birth, 8/9 birthplace steps ask a nationality, over-application 3/34 (v3.0: 2/34)
#   v3.7  the same place, reworded to keep the resolve-first order ("Once the people or things ... are known by
#         name"): v3.6 made the model skip resolving ("When did the director of X die?" instead of "Who directed
#         X?") in 13 of 80 2Wiki first steps; v3.7: 0/80, and 0/29 refusals, 35/35, 9/9, over-application 2/34
_MERGED = ("Once the people or things the question describes are known by name, ask for a fact a document states "
           "directly, of the kind the question asks about: to find out who is older, ask for each person's date of birth, "
           "not how old they are; to find out which country someone is from, ask for their nationality, not their place of birth.")
_RULES = (_MERGED if {"stated", "samekind"} <= FIXES else _STATED if "stated" in FIXES else
          _SAMEKIND if "samekind" in FIXES else "")
_EXNOTE = "\n\n(The example is only an illustration; do not repeat it.)" if "exnote" in FIXES else ""
_RULES_AFTER_EXAMPLE = ("\n" + _RULES + _EXNOTE + "\n") if _RULES else ""     # v3.6: after the example

_HEAD_NOFACTS = _DESC + """

Write ONE short question that finds a piece of information needed to answer the question below. Ask about a description that is not built on another one, so your question can be searched on its own. If the question has no such description""" + (" and needs no calculation" if "stated" in FIXES else "") + """, ask the question itself.

Example
Question: What language is spoken in the birthplace of the author of Y?
Next question: Who is the author of Y?
""" + _RULES_AFTER_EXAMPLE + """
Question: {q}"""

_HEAD_FACTS = "Below is a question, and the facts found so far while working on it.\n\n" + _DESC + """

Write ONE short question that finds the next missing piece of information needed to answer the main question. Use the facts found so far: write names in place of descriptions the facts already resolve. Ask about a description that is not built on another unresolved one, so your question can be searched on its own. Do not ask for anything already in the facts.

Example
Question: What language is spoken in the birthplace of the author of Y?
Facts found so far:
1. Who is the author of Y? Answer: Jane Doe
Next question: Where was Jane Doe born?
""" + _RULES_AFTER_EXAMPLE + """
Question: {q}
Facts found so far:
{facts}"""

_FAILED_NOFACTS = """

These questions could not be answered from the documents:
{failed}
Do not ask any of these again. Ask for the needed information in different words, or start from another part of the question."""

_FAILED_FACTS = """

These questions could not be answered from the documents:
{failed}
Do not ask any of these again. Ask for the needed information in different words, or work on another part of the main question."""

_REPEAT = '\n\nThe question "{sq}" was already asked. Write a different question.'
# fix "resolvefirst": the next sub-question after one that got no answer
# Only the failed question is shown: with the facts shown, gpt-4o-mini put the OTHER item's name into comparison
# questions ("When did the director of Brother Rat, who is not Pierre Montazel, die?"); putting in names the facts
# give is the planner's job (offline check, our development notes).
RESOLVE_T = """The question below could not be answered from the documents.

Question: {sq}

If it refers to a person or thing by a description instead of by name, such as "the author of Y", the documents may need to be searched for that name first. Write ONE short question that finds the name, asking about a description that is not built on another one. If the question refers to everything by name, reply NONE.

Example
Question: When was the author of Y born?
Question that finds the name: Who is the author of Y?

Reply with only the question, or NONE."""

def resolve_prompt(sq):
    return RESOLVE_T.format(sq=sq)
_TAIL = "\n\nReply with only the next question."

def next_prompt(q, facts, failed, repeat_of=None):
    """Builds one of four sub-query prompts:
    SUBQ_NO_FACTS (no facts, no failed), SUBQ_FACTS (facts only),
    SUBQ_NO_FACTS_FAILED (failed only), SUBQ_FACTS_FAILED (facts and failed)."""
    if facts:
        p = _HEAD_FACTS.format(q=q, facts=facts_text(facts))
        if failed: p += _FAILED_FACTS.format(failed="\n".join(f"- {x}" for x in failed))
    else:
        p = _HEAD_NOFACTS.format(q=q)
        if failed: p += _FAILED_NOFACTS.format(failed="\n".join(f"- {x}" for x in failed))
    if repeat_of: p += _REPEAT.format(sq=repeat_of)
    return p + _TAIL

_FINAL_FOCUS = ""
# fix "dates": with correct birth dates the final step compared them wrongly or called the later-born
# person older (2Wiki comparison errors +4/-0; 3/3 such slips fixed).
_FINAL_FOCUS += ("When the question compares dates, compare the years first, then the months, then the days. "
                 "When it compares ages, the person born earlier is older.\n\n" if "dates" in FIXES else "")
# fix "samecountry": "same nationality/country" questions were answered "No" on overlapping facts
# ("French" vs "French-Armenian"). Gated by keyword: in every final prompt the line changed 7 unrelated
# answers for the worse, so it is added only when the question asks about the same country/nationality.
_SAMECOUNTRY_LINE = ("Two people or things are from the same country if they share at least one country, counting a "
                     "city, state or territory as part of its country (for example, 'Greek American' and 'American' "
                     "share the United States).\n\n")
SAMECOUNTRY_Q = re.compile(r"\bsame (nationality|country)\b|\bfrom the same\b|\bshare the same\b|"
                           r"\b(nationality|country) as\b", re.I)

def _final_templates(focus):
    """(FINAL_T, FINAL_RETRY_T, FINAL_REASON_T) with the given focus lines."""
    final = ("Answer the question using ONLY the facts below. Give a short, direct answer -- a few words, not a sentence.\n\n"
             + focus + "Facts:\n{facts}\n\nQuestion: {q}\nAnswer:")
    retry = ("Answer the question using ONLY the facts below. Give a short, direct answer -- a few words, not a sentence.\n\n"
             "An earlier attempt gave no answer. These facts were judged sufficient to answer the question, so do not reply "
             "that the answer is unknown or not stated. Combine the facts to give your best answer.\n\n"
             + focus + "Facts:\n{facts}\n\nQuestion: {q}\nAnswer:")
    reason = ("Answer the question using ONLY the facts below.\n\nFirst work out the answer from the facts in one or two "
              "short lines, doing any comparison or arithmetic the question needs. Then give the answer on a last line "
              "starting with \"Answer:\", in a few words.\n\n" + focus + "Facts:\n{facts}\n\nQuestion: {q}")
    return final, retry, reason

FINAL_T, FINAL_RETRY_T, FINAL_REASON_T = _final_templates(_FINAL_FOCUS)
_FINAL_T_SC, _FINAL_RETRY_T_SC, _FINAL_REASON_T_SC = _final_templates(_FINAL_FOCUS + _SAMECOUNTRY_LINE)

# keyword gate for the "reason" fix: questions that compare lifespans
LIFESPAN_Q = re.compile(r"\blived (longer|shorter|the longest|the shortest)\b|\blonger life\b|\bage at death\b|"
                        r"\bdied (older|younger)\b", re.I)

def _last_answer(text):
    """The text after the last "Answer:" line of a worked-out reply (else its last non-empty line)."""
    m = re.findall(r"(?im)^\s*answer\s*:\s*(.+)$", text or "")
    if m: return m[-1].strip()
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return lines[-1] if lines else ""

def final_from_facts(question, pairs, fin=None):
    """The final answer from the facts, with the keyword-gated fixes (samecountry, reason) and ONE retry
    when empty. Records the intermediate replies in fin. Used by solve() and by the offline replays."""
    fin = {} if fin is None else fin
    sc = "samecountry" in FIXES and SAMECOUNTRY_Q.search(question)
    final_t, retry_t, reason_t = (_FINAL_T_SC, _FINAL_RETRY_T_SC, _FINAL_REASON_T_SC) if sc else \
                                 (FINAL_T, FINAL_RETRY_T, FINAL_REASON_T)
    if "reason" in FIXES and LIFESPAN_Q.search(question):        # lifespan: work it out, answer on the last line
        fin["final_work"] = _llm(reason_t.format(facts=facts_text(pairs), q=question), max_tokens=320)
        raw = _last_answer(fin["final_work"])
    else:
        raw = _llm(final_t.format(facts=facts_text(pairs), q=question))
    fin["final_answer"] = raw
    if empty(raw):                                                # ONE retry, with a changed input
        raw = _llm(retry_t.format(facts=facts_text(pairs), q=question))
        fin["final_retry"] = raw
    return raw

GAP = {"type": "noul",
    "instructions": "Do the facts in `facts` together contain all the information needed to answer `question` -- whether stated outright or derivable by combining them?",
    "criteria": {"true": "Every fact needed is present; the answer is stated or derivable by connecting them.",
                 "false": "At least one fact needed to answer is missing and would have to be found elsewhere."}}

# Gap-closed question for the v2 decision format (fine-tuned StartLux-4B, run 2c on): worded apart
# from SUFFICIENT (the two near-identical questions made gap training shift the sufficient gate, run 2b) and about the
# facts ALREADY answered -- the label is "the answer is already among the facts" (decision_model/build_realgap.py).
GAP_V2 = {"type": "noul",
    "instructions": "Has `question` already been answered by the facts in `facts`: does every entity or value the question asks for, or depends on, appear among the facts' answers?",
    "criteria": {"true": "The facts already give the answer: each step the question depends on is answered in `facts`.",
                 "false": "Some step the question depends on has not been answered yet, so `question` cannot be answered from `facts` alone."}}
GAP_Q = os.environ.get("DEFTRAG_GAP_Q", "v2")   # "v2" (default; the paper): GAP_V2, only with DEFTRAG_DECIDER_FORMAT=v2

# A reply that says nothing counts as empty. A plain "No" is a real answer.
NON_ANSWER = re.compile(r"^\W*(none|n/?a|unknown|no (information|answer)|cannot be determined|"
                        + (r"not applicable|" if "notapplicable" in FIXES else "") +
                        r"not (mentioned|stated|given|found|provided|specified|available|supported)|"
                        r"(information|evidence) (is )?(not (provided|given|available|found|stated|mentioned)|insufficient)|"
                        r"(insufficient|not enough) (information|evidence)|there is no (information|mention)|"
                        r"no passages? (mentions?|states?|contains?|provides?|says?)|"
                        r"(the )?((provided|given) )?passages? do(es)? not)\b", re.I)

def empty(a):
    return not a.strip() or bool(NON_ANSWER.match(a))

def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split())

def facts_text(facts):
    return "\n".join(f"{i}. {q} Answer: {a}" for i, (q, a) in enumerate(facts, 1)) or "(none yet)"

def parse_sq(out, fallback):
    """First non-empty line; strip a leading label and surrounding quotes."""
    line = next((l.strip() for l in out.splitlines() if l.strip()), "")
    line = re.sub(r"^\s*(next question|first question|question)\s*:\s*", "", line, flags=re.I)
    return line.strip().strip('"\'“”‘’').strip() or fallback

def llm_answer(q, P_ids, tmpl=None):
    return _llm((tmpl or ANSWER_T).format(ctx=_ctx(texts(P_ids)), q=q))

def gap_closed(question, pairs):
    a = dm({"model": DM_MODEL, "state": {"question": question, "facts": facts_text(pairs)},
             "questions": {"gap_closed": GAP_V2 if (GAP_Q == "v2" and DECIDER_FORMAT == "v2") else GAP}})
    return a["gap_closed"]["noul"]

def scored(q, k, pre=None, emb=None):
    """(ids, relevance scores); all k passages scored in ONE DM call."""
    if pre: return pre
    ids = retrieve_ids(q, k, emb)
    return ids, relevance(q, texts(ids))

def answer_sub_query(q, pre10=None, emb=None, tmpl=None):
    """Answer q from its filtered top-10. If the answer is empty, retry ONCE with the filtered
    top-20 (skipped when the kept passages do not change: same passages, same answer).
    Returns (answer, kept passage ids, scored top-20 or None, log)."""
    ids10, rel10 = scored(q, TOP_K, pre10, emb)
    P = filter_passages(ids10, rel10)
    raw = llm_answer(q, P, tmpl)
    log = {"first_answer": raw, "max_rel10": round(max(rel10), 2), "n_kept": len(P), "retried": False}
    top20 = None
    if empty(raw):
        top20 = scored(q, WIDE_K, None, emb)
        P20 = filter_passages(*top20)
        log.update({"retried": True, "max_rel20": round(max(top20[1]), 2)})
        if P20 != P:
            P = P20
            raw = llm_answer(q, P, tmpl)
            log.update({"retry_answer": raw, "n_kept": len(P)})
        else:
            log["retry_skipped"] = True
    return raw, P, top20, log

# ---------------- sub-query reuse cache ----------------
def _same_q(i):
    return {"type": "noul",
      "instructions": f"Are `query` and `c{i}` the same question: asking for the same piece of information about the same explicitly named entity, so that one correct answer is correct for both?",
      "criteria": {"true": "Both name the same entity explicitly and ask for the same information; only the wording differs.",
                   "false": "They differ in entity, property, time, place or scope -- OR either one refers to something without naming it (such as \"the tournament\" or \"the author\"), so it is unclear whether they mean the same thing."}}

def same_judge(sq, cand_sqs):
    """One DM call: probability that sq is the same question as each candidate."""
    COUNTERS["cache_same_calls"] += 1
    state = {"query": sq, **{f"c{i+1}": c for i, c in enumerate(cand_sqs)}}
    a = dm({"model": DM_MODEL, "state": state, "questions": {f"s{i+1}": _same_q(i + 1) for i in range(len(cand_sqs))}})
    return [a[f"s{i+1}"]["noul"] for i in range(len(cand_sqs))]

_CACHE = "unset"
_CACHE_BASE = os.path.abspath(os.environ["DEFTRAG_CACHE_DIR"]) if REUSE else None
def get_cache():
    """The on-disk cache for this configuration, or None when reuse is off (DEFTRAG_REUSE != 1)."""
    global _CACHE
    if _CACHE != "unset": return _CACHE
    base = _CACHE_BASE
    if not base:
        _CACHE = None; return None
    import hashlib
    from cache import SubQueryCache
    m = _embedder()
    _v2 = {"decider": DM_MODEL, "decider_format": "v2", "REL_TAU_YN": REL_TAU_YN, "rel_prompt_yn": _rel_yn(1)} if DECIDER_FORMAT == "v2" else {}
    answer_path = json.dumps({**_v2, "ANSWER_T": ANSWER_T, "rel_prompt": _rel_q(1), "REL_TAU": REL_TAU,
        "REL_FLOOR": REL_FLOOR, "WEAK_FLOOR": WEAK_FLOOR, "TOP_K": TOP_K, "WIDE_K": WIDE_K,
        "CAP_PASSAGE": CAP_PASSAGE, "NON_ANSWER": NON_ANSWER.pattern}, sort_keys=True)
    emb_id = hashlib.sha1(m.corpus_embeddings[0].float().numpy().round(5).tobytes()).hexdigest()[:8]
    manifest = {"corpus": CORPUS, "n_passages": len(m.corpus_list), "llm": LLM_MODEL,
                "dm": DM_MODEL, "embedding_dim": int(m.corpus_embeddings.shape[1]), "embedding_id": emb_id,
                "answer_path_sha1": hashlib.sha1(answer_path.encode()).hexdigest()}
    part = f"{DATASET}/{LLM_MODEL.replace(':', '_')}__emb{manifest['embedding_dim']}-{emb_id}__{manifest['answer_path_sha1'][:8]}"
    _CACHE = SubQueryCache(os.path.join(base, part), manifest)
    return _CACHE

def cached_answer(sq, qid, step):
    """Answer a sub-query, reusing the cache when possible. Returns (answer, passage ids, log).
    With no cache this is exactly answer_sub_query."""
    cache = get_cache()
    if cache is None:
        a, P, _, log = answer_sub_query(sq, tmpl=ANSWER_T)
        return a, P, log
    key = norm(sq)
    e = cache.exact(key, qid)
    if e is not None:
        COUNTERS["cache_exact"] += 1
        return e["answer"], e["ids"], {"cache": "exact", "cache_src": e["qid"], "cache_status": e["status"],
                                       "first_answer": e["answer"], "retried": False}
    emb = embed_query(sq)
    cands = cache.candidates(emb, qid, CACHE_K)
    log_c = {}
    if cands:
        ps = same_judge(sq, [c["sq"] for c, _ in cands])
        j = max(range(len(ps)), key=lambda i: ps[i])
        (c, sim), p = cands[j], ps[j]
        log_c = {"cache_best_p": round(p, 3), "cache_best_sim": round(sim, 3), "cache_best_sq": c["sq"]}
        if p >= SAME_TAU:
            COUNTERS["cache_similar"] += 1
            return c["answer"], c["ids"], dict(log_c, cache="similar", cache_src=c["qid"],
                                               first_answer=c["answer"], retried=False)
    a, P, _, log = answer_sub_query(sq, emb=emb, tmpl=ANSWER_T)
    cache.insert(key, sq, "failed" if empty(a) else "answer", a, P, qid, step, emb)
    return a, P, dict(log, cache="miss", **log_c)

# ---------------- the loop ----------------
def solve(question, verbose=False, qid=None, loop_only=False):
    """Answer one question. Returns (answer, trace). loop_only (implied by "nogate") skips the v3.0 gate
    and direct attempt: the loop starts with no facts and no failures."""
    qid = qid or question
    loop_only = loop_only or "nogate" in FIXES
    trace, facts, failed = [], [], []
    original_query_top20 = None

    rec0 = {"step": 0, "query": question, "fixes": sorted(FIXES), "loop_only": loop_only}
    trace.append(rec0)
    if not loop_only:                                     # the gate (v3.0, or stricter with "hybridgate")
        ids10 = retrieve_ids(question)
        suff, depth, rel10 = trigger(question, texts(ids10))
        rec0.update({"sufficient": round(suff, 3), "depth": round(depth, 3)})
        if "hybridgate" in FIXES:
            kin = bool(KIN_Q.search(question)); rec0["kin"] = kin
            go_direct = suff >= GATE_SUFF_HYBRID and depth <= DEPTH_TAU and not kin
        else:
            go_direct = suff >= GATE_SUFF and depth <= DEPTH_TAU
        if go_direct:
            a, P, top20, log = answer_sub_query(question, pre10=(ids10, rel10))   # reuse TRIGGER scores
            rec0["direct"] = dict(log, answer=a)
            if not empty(a):
                rec0.update({"exit": "direct", "answer": a})
                if verbose: print(json.dumps(rec0, ensure_ascii=False))
                return a, trace
            original_query_top20 = top20
    if verbose: print(json.dumps(rec0, ensure_ascii=False))

    sufficient, exit_ = False, "cap"
    just_failed = None                                    # "resolvefirst": the sub-question that just got no answer
    for step in range(1, MAX_STEPS + 1):
        pairs = [(f["SQ"], f["answer"]) for f in facts]
        asked = {norm(x) for x in failed} | {norm(f["SQ"]) for f in facts}
        resolved = None
        if "resolvefirst" in FIXES and just_failed:       # ONE resolve attempt per failed sub-question
            r = parse_sq(_llm(resolve_prompt(just_failed)), "NONE")
            if not re.match(r"^\W*none\b", r, re.I) and norm(r) and norm(r) not in asked:
                resolved = r
        just_failed = None
        SQ = resolved or parse_sq(_llm(next_prompt(question, pairs, failed)), question)
        repeat = norm(SQ) in asked
        if repeat:                                        # ONE regeneration, with a changed input
            first = SQ
            SQ = parse_sq(_llm(next_prompt(question, pairs, failed, repeat_of=first)), question)
            if norm(SQ) in asked:
                trace.append({"step": step, "SQ": SQ, "repeat": True, "repeat_of": first, "stuck": True})
                exit_ = "stuck"
                break

        a, P, log = cached_answer(SQ, qid, step)
        rec = {"step": step, "SQ": SQ, "repeat": repeat, "answer": a,
               "widened": log["retried"], **log}          # "widened" = empty answer retried with top-20
        if resolved: rec["resolved_from"] = failed[-1]
        if empty(a):
            failed.append(SQ)
            rec["accepted"] = False
            just_failed = SQ
        else:
            facts.append({"SQ": SQ, "answer": a, "ids": P})
            g = gap_closed(question, [(f["SQ"], f["answer"]) for f in facts])
            rec.update({"accepted": True, "gap_closed": round(g, 3)})
            if g >= GAP_TAU:
                sufficient, exit_ = True, "gap-closed"
        trace.append(rec)
        if verbose: print(json.dumps(rec, ensure_ascii=False)[:400])
        if sufficient: break

    pairs = [(f["SQ"], f["answer"]) for f in facts]
    fin = {"exit": exit_, "facts": facts, "failed": failed}
    ans = None
    if sufficient:
        raw = final_from_facts(question, pairs, fin)
        if not empty(raw):
            ans, fin["route"] = raw, "facts"
    if ans is None:                                       # facts not sufficient (or both empty)
        # Passages only. Adding the facts from a chain that did not close slightly hurt, in two
        # isolated tests of this final step: passages only 22 vs passages + verified facts 19,
        # of 72 questions, and 24 vs 21 of 82 with unverified facts.
        # Neither difference is significant on its own; both point
        # the same way, and passages only is the simpler choice. "fbfacts" adds the facts ONLY when
        # this passages answer is a non-answer (always graded wrong), so it cannot undo that.
        P = filter_passages(*(original_query_top20 or scored(question, WIDE_K)))
        ans = _llm(ANSWER_T.format(ctx=_ctx(texts(P)), q=question))
        fin.update({"route": "passages", "fallback": True, "n_passages": len(P)})
        if "fbfacts" in FIXES and empty(ans) and pairs:          # ONE retry, with the facts added
            fin["fallback_first"] = ans
            ans = _llm(FALLBACK_FACTS_T.format(facts=facts_text(pairs), ctx=_ctx(texts(P)), q=question))
            fin["route"] = "passages+facts"
    fin["answer"] = ans
    trace.append(fin)
    return ans, trace

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "In what county is the city that borders the capital of the state whose team won the 2017 NCAA women's basketball tournament?"
    ans, tr = solve(q, verbose=True)
    print("\n=== TRACE ===")
    for r in tr: print(json.dumps(r, ensure_ascii=False))
    print("\nANSWER:", ans)
