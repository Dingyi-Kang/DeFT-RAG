#!/usr/bin/env python3
"""Inputs for running DeFT-RAG on TRAINING-split questions (to collect realistic gap-closed states for fine-tuning).
Per dataset: 1,000 training questions (gold paragraph not in any test corpus; hop / type mix of our test set), their
corpus = the union of their paragraphs (deduplicated up to spacing; every test-corpus passage excluded) -- the same
construction as our test corpora (1,000 dev questions each) -- and a run list of N of them (stratified the same way).
Writes LogicRAG/dataset/<name>_corpus.json (DeFT-RAG reads corpora there; DEFTRAG_CORPUS=<name>_corpus.json) and
decision_model/trainrun/<name>_questions.json (run_eval.py qfile mode), plus the gold hop answers for labelling.
usage: make_train_runs.py"""
import os, sys, json, re, random, glob, collections
R = os.path.abspath(os.environ.get("DEFTRAG_WORK", "work")); DATA = os.path.abspath(os.path.expanduser(os.environ.get("DEFTRAG_DATA", "data"))); LG = os.path.join(os.path.abspath(os.path.expanduser(os.environ.get("LOGICRAG_DIR", "LogicRAG"))), "dataset")
OUT = f"{R}/trainrun"; os.makedirs(OUT, exist_ok=True)
rng = random.Random(7)
TESTP = set()
for f in ("musique", "2wikimultihopqa", "hotpotqa"):
    TESTP |= {(c["title"], re.sub(r"\s+", "", c["text"])) for c in json.load(open(f"{LG}/{f}_corpus.json"))}
nk = lambda t, x: (t, re.sub(r"\s+", "", x))
def stratified(rows, keyf, mix, n):
    by = collections.defaultdict(list)
    for r in rows: by[keyf(r)].append(r)
    tot = sum(mix.values()); out = []
    for k, c in mix.items(): pool = by.get(k, []); rng.shuffle(pool); out += pool[:round(n * c / tot)]
    rng.shuffle(out); return out
def write(name, group, paras_of, run_n, keyf, mix, meta):
    seen = {}; 
    for r in group:
        for t, x in paras_of(r):
            if nk(t, x) not in TESTP and nk(t, x) not in seen: seen[nk(t, x)] = {"title": t, "text": x}
    json.dump(list(seen.values()), open(f"{LG}/{name}_corpus.json", "w"), ensure_ascii=False)
    run = stratified(group, keyf, mix, run_n)
    json.dump([{"id": r["id"], "hop": keyf(r), "question": r["question"].strip(), "gold": r["answer"]} for r in run],
              open(f"{OUT}/{name}_questions.json", "w"), ensure_ascii=False)
    json.dump({r["id"]: meta(r) for r in run}, open(f"{OUT}/{name}_gold.json", "w"), ensure_ascii=False)
    print(f"{name}: corpus {len(seen)} passages from {len(group)} questions; run list {len(run)}", dict(collections.Counter(keyf(r) for r in run)))
# MuSiQue
test = json.load(open(f"{R}/GraphRAG-Benchmark/Datasets/Questions/musique_questions.json"))
mix = collections.Counter(q["id"].split("-")[1].split("__")[0] for q in test)
tr = [json.loads(l) for l in open(f"{DATA}/musique/musique_ans_v1.0_train.jsonl")]
def mq_gold(r):
    gi = {h["paragraph_support_idx"] for h in r["question_decomposition"]}
    return {nk(p["title"], p["paragraph_text"]) for p in r["paragraphs"] if p["idx"] in gi}
tr = [r for r in tr if not (mq_gold(r) & TESTP)]
g = stratified(tr, lambda r: r["id"].split("__")[0], mix, 1000)
write("musique_trainrun", g, lambda r: [(p["title"], p["paragraph_text"]) for p in r["paragraphs"]], 800,
      lambda r: r["id"].split("__")[0], mix,
      lambda r: {"hops": [h["answer"] for h in r["question_decomposition"]], "finals": [r["answer"]] + list(r.get("answer_aliases") or [])})
# 2Wiki
import pyarrow.parquet as pq
test = json.load(open(f"{R}/GraphRAG-Benchmark/Datasets/Questions/2wikimultihopqa_questions.json"))
mix = collections.Counter(q["question_type"] for q in test)
rows = [x for f in sorted(glob.glob(f"{DATA}/2wiki/train-*.parquet")) for x in pq.read_table(f).to_pylist()]
paras = lambda r: [(t, " ".join(ss)) for t, ss in zip(r["context"]["title"], r["context"]["sentences"])]
gold2 = lambda r: {nk(t, x) for t, x in paras(r) if t in set(r["supporting_facts"]["title"])}
rows = [r for r in rows if gold2(r) and not (gold2(r) & TESTP) and r["evidences"]]
g = stratified(rows, lambda r: r["type"], mix, 1000)
write("2wiki_trainrun", g, paras, 500, lambda r: r["type"], mix,
      lambda r: {"hops": [e[2] for e in r["evidences"]], "finals": [r["answer"]]})
