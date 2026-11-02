#!/usr/bin/env python3
"""Exact match (EM) and token F1 of a run, as reported in the paper (no model is called).

usage: em_f1.py BENCH.json [BENCH.json ...]      (runs converted with to_bench.py)

SQuAD-style normalisation (lower case, drop punctuation and articles, collapse spaces). A question's score is the
maximum over its gold answer and, for MuSiQue, the answer aliases in $LOGICRAG_DIR/dataset/musique.json. Answers are
scored as given; nothing is cut or cleaned beyond the normalisation.
"""
import json, os, re, string, collections, sys
LG = os.path.abspath(os.path.expanduser(os.environ.get("LOGICRAG_DIR", "LogicRAG")))
def norm(s):
    s = str(s).lower(); s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def f1(p, g):
    p, g = norm(p).split(), norm(g).split()
    c = sum((collections.Counter(p) & collections.Counter(g)).values())
    if not p or not g: return float(p == g)
    if c == 0: return 0.0
    pr, rc = c / len(p), c / len(g); return 2 * pr * rc / (pr + rc)
ALIAS = {}
for r in json.load(open(os.path.join(LG, "dataset", "musique.json"))):
    ALIAS[norm(r["question"])] = [r["answer"]] + list(r.get("answer_aliases") or [])
def golds(r):
    return ALIAS.get(norm(r.get("question", "")), [r["ground_truth"]]) if r.get("ground_truth") is not None else []
def score(path):
    R = json.load(open(path)); em = f = 0
    for r in R:
        gs = golds(r) or [""]; a = r.get("generated_answer") or ""
        em += max(float(norm(a) == norm(g)) for g in gs); f += max(f1(a, g) for g in gs)
    return em / len(R), f / len(R), len(R)
if __name__ == "__main__":
    for p in sys.argv[1:]:
        e, f, n = score(p); print(f"{os.path.basename(p)}: n={n} EM={e:.3f} F1={f:.3f}")
