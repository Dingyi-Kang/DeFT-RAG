#!/usr/bin/env python3
"""Fine-tuning data for the decision model in DeFT-RAG, from the TRAINING splits of MuSiQue,
2WikiMultihopQA and HotpotQA, labelled ONLY by their gold annotations (no model outputs, no test questions; our test
questions are all from the dev splits -- verified, 0 overlap with train).

RETRIEVAL MIRRORS THE TEST SETUP. Our test corpora are the paragraphs of 1,000 dev questions per benchmark. Here the
sampled training questions are split into GROUPS OF 1,000; each group's corpus is the union of its questions' own
paragraphs (deduplicated by exact title+text; passages that occur in any TEST corpus excluded). Every request retrieves
its top 10 from its group's whole corpus with DeFT-RAG's embedder and query instruction -- the main question and each
sub-question alike. Gold = the annotated supporting paragraph (exact title+text), the same rule as the test labels.

Requests, shaped exactly like DeFT-RAG's (question definitions imported from loop):
  trigger  state {question, p1..p10}; sufficient (U.SUFFICIENT) + rel1..rel10 (U._rel_yn, yes/no)
           rel = gold supporting paragraph; sufficient = all gold supporting paragraphs are among p1..p10
  subq     state {question: a self-contained MuSiQue sub-question, p1..p10}; rel1..rel10; gold = that hop's paragraph
  depth    state {question} only; depth (U.DEPTH); label = hops - 1                                          [MuSiQue]
  gap      state {question, facts: U.facts_text(...)}; gap_closed (U.GAP); label = every gold hop answer is among
           the facts (the test-set rule match, the test-set rule)                                    [MuSiQue]
Passages: "Title: {title}. Content: {text}" (loop.ptext; capped at U.CAP_PASSAGE for the decision model,
uncapped for embedding, as BaseRAG). Paragraph text joined as our corpora were (verified: 2Wiki ' '.join(sentences),
HotpotQA ''.join(sentences), MuSiQue paragraph_text).
Sampling: fixed seed; hop / type mix matched to our 1,000-question test sets. Train / dev = 85 / 15 by question id.
Questions whose GOLD paragraph occurs in a test corpus are skipped.
Output: OUT/{train,dev}.jsonl in StartLux's fine-tuning format {state, questions, targets, meta} + OUT/report.json.
usage: build_ft_data.py OUT_DIR [N_PER_DATASET=3000]"""
import os, sys, json, re, random, glob, collections, hashlib
import numpy as np
R = os.path.abspath(os.environ.get("DEFTRAG_WORK", "work")); DATA = os.path.abspath(os.path.expanduser(os.environ.get("DEFTRAG_DATA", "data")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deftrag"))
import loop as U                                   # question definitions, facts_text, norm, CAP_PASSAGE (no calls)
from retriever import Embedder, QUERY_INSTRUCTION      # DeFT-RAG's retrieval embedder
OUT = os.path.abspath(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
os.makedirs(OUT, exist_ok=True)
SEED, DEV_FRAC, K, GROUP = 42, 0.15, 10, 1000
QI = "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: "   # = retriever.QUERY_INSTRUCTION
assert QI == QUERY_INSTRUCTION, "query instruction differs from DeFT-RAG retrieval"
rng = random.Random(SEED)
rep = collections.defaultdict(collections.Counter)
TESTP = set()                                            # (title, text) of every passage in our TEST corpora
for _f in ("musique", "2wikimultihopqa", "hotpotqa"):
    TESTP |= {(c["title"], c["text"]) for c in json.load(open(os.path.join(os.path.abspath(os.path.expanduser(os.environ.get("LOGICRAG_DIR", "LogicRAG"))), f"dataset/{_f}_corpus.json")))}

# ---------------- embedding (same model, pooling, instruction framing as DeFT-RAG's retrieval) ----------------
_E = None; _cache = {}
CACHE = f"{OUT}/emb_cache.npz"
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True); _cache = dict(zip(z["k"], z["v"]))
def _key(s, q): return hashlib.md5((("Q|" if q else "P|") + s).encode()).hexdigest()
def embed(texts, query=False):
    global _E
    ks = [_key(t, query) for t in texts]; need = [t for t, k in zip(texts, ks) if k not in _cache]
    if need:
        if _E is None: _E = Embedder()
        uniq = list(dict.fromkeys(need))
        for t, v in zip(uniq, _E.encode(uniq, is_query=query).numpy()): _cache[_key(t, query)] = np.asarray(v, dtype=np.float32)
    V = np.stack([_cache[k] for k in ks]); return V / np.linalg.norm(V, axis=1, keepdims=True)
def save_cache(): np.savez(CACHE, k=np.array(list(_cache), dtype=object), v=np.stack(list(_cache.values())))
def ptext(title, text): return f"Title: {title}. Content: {text}"
def nkey(p):                                            # same passage up to spacing: 2Wiki has variants like
    return (p[0], re.sub(r"\s+", "", p[1]))              # "Carlmar( Edith" vs "Carlmar (Edith" of one paragraph

class Corpus:
    """One group's corpus: the deduplicated paragraphs of its questions, minus any passage of a test corpus."""
    def __init__(self, passages):
        first = {}                                       # dedup up to spacing (nkey), first variant kept
        for p in passages:
            if nkey(p) in first: rep["dedup"]["spacing / exact duplicates merged"] += first[nkey(p)] != p
            else: first[nkey(p)] = p
        self.p = list(first.values())
        self.V = embed([ptext(t, x) for t, x in self.p]); save_cache()
    def top(self, question, k=K):
        q = embed([question], True)[0]
        return [self.p[i] for i in np.argsort(-(self.V @ q), kind="stable")[:k]]

def rel_request(question, top, gold, extra=None):
    state = {"question": question, **{f"p{i+1}": ptext(t, x)[:U.CAP_PASSAGE] for i, (t, x) in enumerate(top)}}
    qs = {f"rel{i}": U._rel_yn(i) for i in range(1, len(top) + 1)}
    gk = {nkey(g) for g in gold}
    tg = {f"rel{i}": {"true" if nkey(p) in gk else "false": 1.0} for i, p in enumerate(top, 1)}
    if extra: qs.update(extra[0]); tg.update(extra[1])
    return state, qs, tg
def trigger_request(question, top, gold):
    suff = bool(gold) and {nkey(g) for g in gold} <= {nkey(p) for p in top}
    return rel_request(question, top, gold, ({"sufficient": U.SUFFICIENT}, {"sufficient": {"true" if suff else "false": 1.0}})), suff
def match(a, gs):                                          # = the test-set rule match() (the test-set gap rule)
    a = U.norm(a); return bool(a) and any(len(g) >= 2 and (g in a or (len(a) >= 3 and a in g)) for g in map(U.norm, gs))
def covered(facts, hops, finals):
    return all(any(match(a, [g] if j < len(hops) - 1 else [g] + finals) for _, a in facts) for j, g in enumerate(hops))
def stratified(rows, keyf, target_mix, n):
    by = collections.defaultdict(list)
    for r in rows: by[keyf(r)].append(r)
    tot = sum(target_mix.values()); out = []
    for k, c in target_mix.items():
        want = round(n * c / tot); pool = by.get(k, [])
        rng.shuffle(pool); out += pool[:want]
        rep["sampling"][f"{k}: wanted {want}, available {len(pool)}"] += 1
    rng.shuffle(out); return out
def is_dev(qid): return int(hashlib.md5(f"{SEED}:{qid}".encode()).hexdigest(), 16) % 1000 < DEV_FRAC * 1000
def groups(rows): return [rows[i:i + GROUP] for i in range(0, len(rows), GROUP)]

items = {"train": [], "dev": []}
def emit(ds, kind, qid, state, qs, tg):
    part = "dev" if is_dev(qid) else "train"
    items[part].append({"state": state, "questions": qs, "targets": tg, "meta": {"dataset": ds, "kind": kind, "qid": qid}})
    rep["items"][f"{ds}/{kind}/{part}"] += 1

# ================= MuSiQue =================
test_mq = json.load(open(f"{R}/GraphRAG-Benchmark/Datasets/Questions/musique_questions.json"))
mix = collections.Counter(q["id"].split("-")[1].split("__")[0] for q in test_mq)
tr = [json.loads(l) for l in open(f"{DATA}/musique/musique_ans_v1.0_train.jsonl")]
def gold_paras(r):
    gi = {h["paragraph_support_idx"] for h in r["question_decomposition"]}
    return {(p["title"], p["paragraph_text"]) for p in r["paragraphs"] if p["idx"] in gi}
n0 = len(tr); tr = [r for r in tr if not (gold_paras(r) & TESTP)]
rep["musique"]["train questions skipped: gold paragraph in a test corpus"] = n0 - len(tr)
def resolve(r):
    """Sub-questions with #k replaced by hop k's answer; None where a sub-question is not natural language
    ('A >> relation' shorthand) or refers to anything but an EARLIER hop."""
    out, ans = [], [h["answer"] for h in r["question_decomposition"]]
    for j, h in enumerate(r["question_decomposition"]):
        refs = [int(x) for x in re.findall(r"#(\d+)", h["question"])]
        if ">>" in h["question"] or any(not 1 <= k <= j for k in refs): out.append(None); continue
        q = re.sub(r"#(\d+)", lambda m: ans[int(m.group(1)) - 1], h["question"])
        out.append(q if "#" not in q else None)
    return out
sel = stratified(tr, lambda r: r["id"].split("__")[0], mix, N)
for g in groups(sel):
    C = Corpus([(p["title"], p["paragraph_text"]) for r in g for p in r["paragraphs"] if (p["title"], p["paragraph_text"]) not in TESTP])
    rep["musique_corpus"][f"group of {len(g)}: {len(C.p)} passages"] += 1
    for r in g:
        gold = gold_paras(r)
        (state, qs, tg), suff = trigger_request(r["question"], C.top(r["question"]), gold)
        rep["musique"][f"sufficient={suff}"] += 1
        emit("musique", "trigger", r["id"], state, qs, tg)
        emit("musique", "depth", r["id"], {"question": r["question"]}, {"depth": U.DEPTH},
             {"depth": {str(min(3, len(r["question_decomposition"]) - 1)): 1.0}})
        sq = resolve(r); hops_ok = [j for j, q in enumerate(sq) if q]
        if not hops_ok: rep["musique"]["subq skipped: no natural sub-question"] += 1; continue
        j = rng.choice(hops_ok); h = r["question_decomposition"][j]
        gj = {(p["title"], p["paragraph_text"]) for p in r["paragraphs"] if p["idx"] == h["paragraph_support_idx"]}
        state, qs, tg = rel_request(sq[j], C.top(sq[j]), gj)
        rep["musique"][f"subq gold in top10={any('true' in v for v in tg.values())}"] += 1
        emit("musique", "subq", r["id"], state, qs, tg)
    save_cache()
# gap items: only questions whose sub-questions are ALL natural (the fact list shows every one); no retrieval involved
answers_by_hoppos = collections.defaultdict(list)          # wrong answers: other questions' answers at the same hop position
for r in tr:
    for j, h in enumerate(r["question_decomposition"]): answers_by_hoppos[j].append(h["answer"])
nat = [r for r in tr if all(resolve(r))]
for r in stratified(nat, lambda r: r["id"].split("__")[0], mix, N):
    sq = resolve(r); hops = [h["answer"] for h in r["question_decomposition"]]
    finals = [r["answer"]] + list(r.get("answer_aliases") or []); full = list(zip(sq, hops)); n = len(full)
    cands = [("prefix", full[:rng.randint(1, n - 1)]), ("full", full)]
    j = rng.randrange(n); bad = list(full); bad[j] = (sq[j], rng.choice(answers_by_hoppos[j])); cands.append(("wrong_answer", bad))
    o = rng.choice(nat)
    if o["id"] != r["id"]:
        k2 = rng.randrange(len(o["question_decomposition"]))
        plus = list(full); plus.insert(rng.randrange(n + 1), (resolve(o)[k2], o["question_decomposition"][k2]["answer"]))
        cands.append(("full_plus_unrelated_fact", plus))
    for kind, facts in cands:
        lab = covered(facts, hops, finals); intended = kind in ("full", "full_plus_unrelated_fact")
        if lab != intended: rep["musique"][f"gap {kind}: dropped (rule label {lab} != intended)"] += 1; continue
        emit("musique", f"gap_{kind}", r["id"], {"question": r["question"], "facts": U.facts_text(facts)},
             {"gap_closed": U.GAP}, {"gap_closed": {"true" if lab else "false": 1.0}})

# ================= 2Wiki / HotpotQA =================
import pyarrow.parquet as pq
for ds, d, joiner in (("2wikimultihopqa", "2wiki", " "), ("hotpotqa", "hotpotqa", "")):
    test = json.load(open(f"{R}/GraphRAG-Benchmark/Datasets/Questions/{ds}_questions.json"))
    tmix = collections.Counter(q["question_type"] for q in test)
    rows = [x for f in sorted(glob.glob(f"{DATA}/{d}/train-*.parquet")) for x in pq.read_table(f).to_pylist()]
    tkeys = collections.Counter(r["type"] for r in rows)
    rep[ds]["test types"] = dict(tmix); rep[ds]["train types"] = dict(tkeys)
    paras = lambda r: [(t, joiner.join(ss)) for t, ss in zip(r["context"]["title"], r["context"]["sentences"])]
    gold_of = lambda r: {p for p in paras(r) if p[0] in set(r["supporting_facts"]["title"])}
    n0 = len(rows); rows = [r for r in rows if gold_of(r) and not (gold_of(r) & TESTP)]
    rep[ds]["train questions skipped: no gold paragraph, or gold paragraph in a test corpus"] = n0 - len(rows)
    sel = stratified(rows, lambda r: r["type"], {k: v for k, v in tmix.items() if k in tkeys}, N)
    for g in groups(sel):
        C = Corpus([p for r in g for p in paras(r) if p not in TESTP])
        rep[f"{ds}_corpus"][f"group of {len(g)}: {len(C.p)} passages"] += 1
        for r in g:
            (state, qs, tg), suff = trigger_request(r["question"], C.top(r["question"]), gold_of(r))
            rep[ds][f"sufficient={suff}"] += 1
            emit(ds, "trigger", r["id"], state, qs, tg)
        save_cache()

# ---------------- checks and report ----------------
for part in ("train", "dev"):                              # near-duplicate label noise (same title, one gold one not)
    for it in items[part]:
        ps = {k: v for k, v in it["state"].items() if k.startswith("p") and k[1:].isdigit()}
        gt = {ps[k].split(". Content:")[0] for k in ps if "true" in it["targets"].get("rel" + k[1:], {})}
        if any("false" in it["targets"].get("rel" + k[1:], {}) and ps[k].split(". Content:")[0] in gt for k in ps):
            rep["label_noise"][f"{it['meta']['dataset']}/{it['meta']['kind']}: a non-gold passage shares a gold passage's title"] += 1
seen = {v for part in items.values() for it in part for k, v in it["state"].items() if k.startswith("p") and k[1:].isdigit()}
rep["overlap"]["distinct passages shown"] = len(seen)
rep["overlap"]["of them in a test corpus"] = len(seen & {ptext(t, x)[:U.CAP_PASSAGE] for t, x in TESTP})
for part in ("train", "dev"):
    rng.shuffle(items[part])
    with open(f"{OUT}/{part}.jsonl", "w") as f:
        for it in items[part]: f.write(json.dumps(it, ensure_ascii=False) + "\n")
json.dump({k: dict(v) for k, v in rep.items()}, open(f"{OUT}/report.json", "w"), indent=1)
print(json.dumps({k: dict(v) for k, v in rep.items()}, indent=1))
