#!/usr/bin/env python3
"""2Wiki LOOP training items for run 2c: run 2's data had no 2Wiki sub-question or gap-closed items,
and on the full 2Wiki benchmark the fine-tuned model closed the gap on wrong family relations (e.g. a mother's FATHER
for "maternal grandmother"). Built ONLY from the 2Wiki TRAINING split's evidence triples (subject, relation, object),
with FIXED templates (no generated text), for the same 2Wiki training questions as run 2 (data_v2).
  subq  one triple per question -> templated sub-question ("Who is the mother of {s}?"), retrieved top 10 from the
        question's group corpus (groups of 1,000, as build_ft_data.py; DeFT-RAG's embedder and instruction); gold = the
        question's supporting paragraph whose title is the triple's subject. Labels: gold + 3 random non-gold (run 2).
  gap   facts = templated (sub-question, object) per triple; variants: prefix (first k hops) -> no; full -> yes;
        wrong_relation (one hop asked with its counterpart relation, e.g. father for mother, answered with another
        question's object of that relation) -> no; wrong_answer (one hop's object replaced) -> no; full + an unrelated
        fact -> yes. Every label re-checked with the 2Wiki test rule (every evidence object among the facts' answers,
        prep_eval.match); disagreements dropped. Question wording: U.GAP_V2.
Train / dev by the build_ft_data.py question-id hash; ~half the questions kept (select_rows.py hash), as run 2.
usage: build_2wiki_loop.py OUT_DIR"""
import os, sys, json, re, random, glob, hashlib, collections
import numpy as np
R = os.path.abspath(os.environ.get("DEFTRAG_WORK", "work")); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deftrag"))
import loop as U
from retriever import Embedder, QUERY_INSTRUCTION      # DeFT-RAG's retrieval embedder
import pyarrow.parquet as pq
rng = random.Random(11); st = collections.Counter()
QI = "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: "
T = {"director": "Who directed {s}?", "country of citizenship": "What is the nationality of {s}?", "date of birth": "When was {s} born?",
     "country": "Which country is {s} in?", "date of death": "When did {s} die?", "country of origin": "Which country is {s} from?",
     "place of birth": "Where was {s} born?", "father": "Who is the father of {s}?", "publication date": "When was {s} released?",
     "place of death": "Where did {s} die?", "performer": "Who performed {s}?", "spouse": "Who is the spouse of {s}?",
     "composer": "Who composed the music for {s}?", "mother": "Who is the mother of {s}?", "cause of death": "What was the cause of death of {s}?",
     "award received": "What award did {s} receive?", "educated at": "Where was {s} educated?", "child": "Who is a child of {s}?",
     "place of burial": "Where is {s} buried?", "inception": "When was {s} founded?", "occupation": "What is the occupation of {s}?",
     "founded by": "Who founded {s}?", "producer": "Who produced {s}?", "employer": "Who is the employer of {s}?",
     "sibling": "Who is a sibling of {s}?", "publisher": "Who published {s}?", "creator": "Who created {s}?", "editor": "Who is the editor of {s}?",
     "presenter": "Who presented {s}?", "has part": "What is part of {s}?", "place of detention": "Where was {s} detained?",
     "student of": "Whose student was {s}?", "doctoral advisor": "Who was the doctoral advisor of {s}?", "manufacturer": "Who manufactured {s}?"}
COUNTER = {"father": "mother", "mother": "father", "spouse": "sibling", "sibling": "spouse", "child": "sibling",
           "date of birth": "date of death", "date of death": "date of birth", "place of birth": "place of death",
           "place of death": "place of birth", "director": "producer", "producer": "director", "composer": "performer",
           "performer": "composer", "country of citizenship": "place of birth"}
def norm(s): return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split())
def match(a, gs):
    a = norm(a); return bool(a) and any(len(g) >= 2 and (g in a or (len(a) >= 3 and a in g)) for g in map(norm, gs))
nk = lambda t, x: (t, re.sub(r"\s+", "", x))
SEED, DEV_FRAC = 42, 0.15
is_dev = lambda qid: int(hashlib.md5(f"{SEED}:{qid}".encode()).hexdigest(), 16) % 1000 < DEV_FRAC * 1000
keep_q = lambda qid: int(hashlib.md5(f"select:2wikimultihopqa:{qid}".encode()).hexdigest(), 16) % 1000 < 500
TESTP = set()
for f in ("musique", "2wikimultihopqa", "hotpotqa"):
    TESTP |= {nk(c["title"], c["text"]) for c in json.load(open(os.path.join(os.path.abspath(os.path.expanduser(os.environ.get("LOGICRAG_DIR", "LogicRAG"))), f"dataset/{f}_corpus.json")))}
# the 2Wiki training questions run 2 used (their ids in data_v2)
qids = sorted({json.loads(l)["meta"]["qid"] for p in ("train", "dev") for l in open(f"{R}/decision_model/data_v2/{p}.jsonl") if '"2wikimultihopqa"' in l})
rows = {r["id"]: r for f in sorted(glob.glob(os.path.join(os.path.abspath(os.path.expanduser(os.environ.get("DEFTRAG_DATA", "data"))), "2wiki/train-*.parquet"))) for r in pq.read_table(f).to_pylist()}
Q = [rows[q] for q in qids if q in rows and rows[q]["evidences"]]
st["2wiki questions (run 2's)"] = len(qids); st["with evidence triples"] = len(Q)
pool = collections.defaultdict(list)
for r in rows.values():
    for e in r["evidences"]: pool[e[1]].append(e[2])
# embeddings (reuse run 2's cache: same passages)
cache = dict(zip(*(lambda z: (z["k"], z["v"]))(np.load(f"{R}/decision_model/data_v2/emb_cache.npz", allow_pickle=True))))
_E = None
def key(s, q): return hashlib.md5((("Q|" if q else "P|") + s).encode()).hexdigest()
def embed(texts, query=False):
    global _E
    need = [t for t in texts if key(t, query) not in cache]
    if need:
        if _E is None: _E = Embedder()
        for t, v in zip(list(dict.fromkeys(need)), _E.encode(list(dict.fromkeys(need)), is_query=query).numpy()): cache[key(t, query)] = np.asarray(v, np.float32)
    V = np.stack([cache[key(t, query)] for t in texts]); return V / np.linalg.norm(V, axis=1, keepdims=True)
ptext = lambda t, x: f"Title: {t}. Content: {x}"
paras = lambda r: [(t, " ".join(ss)) for t, ss in zip(r["context"]["title"], r["context"]["sentences"])]
items = {"train": [], "dev": []}
def emit(r, kind, state, qs, tg):
    part = "dev" if is_dev(r["id"]) else "train"
    items[part].append({"state": state, "questions": qs, "targets": tg, "meta": {"dataset": "2wikimultihopqa", "kind": kind, "qid": r["id"]}}); st[f"{kind}/{part}"] += 1
Q.sort(key=lambda r: hashlib.md5(r["id"].encode()).hexdigest())
for gi in range(0, len(Q), 1000):
    G = Q[gi: gi + 1000]
    first = {}
    for r in G:
        for p in paras(r):
            if nk(*p) not in TESTP and nk(*p) not in first: first[nk(*p)] = p
    P = list(first.values()); V = embed([ptext(t, x) for t, x in P])
    for r in G:
        if not keep_q(r["id"]): continue
        ev = r["evidences"]
        # ---- sub-question relevance ----
        j = rng.randrange(len(ev)); s, rel, o = ev[j]
        gold_t = [t for t in set(r["supporting_facts"]["title"]) if norm(t) == norm(s)] or [t for t in set(r["supporting_facts"]["title"]) if norm(s) in norm(t) or norm(t) in norm(s)]
        if rel in T and len(gold_t) == 1:
            gold = {nk(t, x) for t, x in paras(r) if t == gold_t[0]}
            sq = T[rel].format(s=s); qv = embed([sq], True)[0]
            top = [P[i] for i in np.argsort(-(V @ qv), kind="stable")[:10]]
            lab = [nk(*p) in gold for p in top]; non = [i for i, y in enumerate(lab) if not y]
            keep = {i for i, y in enumerate(lab) if y} | set(rng.sample(non, min(3, len(non))))
            state = {"question": sq, **{f"p{i+1}": ptext(*p)[:U.CAP_PASSAGE] for i, p in enumerate(top)}}
            qs = {f"rel{i+1}": U._rel_yn(i + 1) for i in range(len(top))}
            tg = {f"rel{i+1}": {"true" if lab[i] else "false": 1.0} for i in keep}
            st[f"subq gold in top10={any(lab)}"] += 1
            emit(r, "subq_2wiki", state, qs, tg)
        else: st["subq skipped (relation untemplated or subject page ambiguous)"] += 1
        # ---- gap closed (clean, templated) ----
        if not all(e[1] in T for e in ev): st["gap skipped: untemplated relation"] += 1; continue
        full = [(T[e[1]].format(s=e[0]), e[2]) for e in ev]; objs = [e[2] for e in ev]; n = len(full)
        cands = [("full", full)]
        if n >= 2: cands.append(("prefix", full[: rng.randint(1, n - 1)]))
        jj = [k for k in range(n) if ev[k][1] in COUNTER]
        if jj:
            k = rng.choice(jj); cr = COUNTER[ev[k][1]]; bad = list(full); bad[k] = (T[cr].format(s=ev[k][0]), rng.choice(pool[cr])); cands.append(("wrong_relation", bad))
        k = rng.randrange(n); bad = list(full); bad[k] = (full[k][0], rng.choice(pool[ev[k][1]])); cands.append(("wrong_answer", bad))
        o2 = rows[rng.choice(qids)] if qids else None
        if o2 and o2["id"] != r["id"] and o2["evidences"] and o2["evidences"][0][1] in T:
            e2 = o2["evidences"][0]; plus = list(full); plus.insert(rng.randrange(n + 1), (T[e2[1]].format(s=e2[0]), e2[2])); cands.append(("full_plus_unrelated_fact", plus))
        for kind, facts in cands:
            labl = all(any(match(a, [ob]) for _, a in facts) for ob in objs); intended = kind in ("full", "full_plus_unrelated_fact")
            if labl != intended: st[f"gap {kind}: dropped (rule {labl} != intended)"] += 1; continue
            emit(r, f"gap2w_{kind}", {"question": r["question"], "facts": U.facts_text(facts)}, {"gap_closed": U.GAP_V2},
                 {"gap_closed": {"true" if labl else "false": 1.0}})
for part in items:
    with open(f"{OUT}/2wiki_loop_{part}.jsonl", "w") as f:
        for x in items[part]: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(json.dumps(dict(st), indent=1))
