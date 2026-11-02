#!/usr/bin/env python3
"""Realistic gap-closed training items from DeFT-RAG runs on TRAINING-split questions (run_eval.py in qfile mode).
At every gap check of a run (accepted steps, as the test-set rule), the state is the run's own question and
fact table; the label is the test-set rule: every gold hop answer is among the facts' answers (prep_eval.match; MuSiQue:
decomposition answers, last hop with the answer aliases; 2Wiki: evidence-triple objects). Train / dev by the same
question-id hash as build_ft_data.py (so dev questions stay held out across all item kinds). No test data.
LABEL (--rule): "answer" (default) = closed when the final gold answer (or an alias) is already among the facts, or every
gold hop is covered -- on real runs the strict rule alone marks bypassed hops ("Italy -> Mazzini" without "Libya") and
paraphrases ("German" vs "Germany") as incomplete although the question is answerable (88 of 345 MuSiQue closes);
"hops" = the strict test-set rule (every gold hop answer among the facts).
usage: build_realgap.py OUT_DIR [--rule answer|hops]"""
import os, sys, json, re, hashlib, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deftrag")); import loop as U
RULE = sys.argv[sys.argv.index("--rule") + 1] if "--rule" in sys.argv else "answer"
T = os.path.join(os.path.abspath(os.environ.get("DEFTRAG_WORK", "work")), "trainrun"); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
SEED, DEV_FRAC = 42, 0.15
is_dev = lambda qid: int(hashlib.md5(f"{SEED}:{qid}".encode()).hexdigest(), 16) % 1000 < DEV_FRAC * 1000
def match(a, gs):
    a = U.norm(a); return bool(a) and any(len(g) >= 2 and (g in a or (len(a) >= 3 and a in g)) for g in map(U.norm, gs))
st = collections.Counter(); out = {"train": [], "dev": []}
for ds, name in (("musique", "musique_trainrun"), ("2wikimultihopqa", "2wiki_trainrun")):
    p = f"{T}/{name}_runs.jsonl"
    if not os.path.exists(p): st[f"{name}: no runs file"] += 1; continue
    gold = json.load(open(f"{T}/{name}_gold.json"))
    for r in map(json.loads, open(p)):
        if r.get("error") or "trace" not in r: st[f"{ds}: run with error, skipped"] += 1; continue
        g = gold[r["id"]]; hops, finals = g["hops"], g["finals"]
        facts = []
        for t in r["trace"][1:]:
            if "SQ" not in t or t.get("accepted") not in (True, "True"): continue
            facts.append((t["SQ"], t["answer"]))
            if t.get("gap_closed") in (None, ""): continue
            if ds == "musique":
                cov = all(any(match(a, [h] if j < len(hops) - 1 else [h] + finals) for _, a in facts) for j, h in enumerate(hops))
            else:
                cov = all(any(match(a, [h]) for _, a in facts) for h in hops)
            if RULE == "answer" and not cov: cov = any(match(a, finals) for _, a in facts)
            part = "dev" if is_dev(r["id"]) else "train"
            out[part].append({"state": {"question": r["question"], "facts": U.facts_text(facts)}, "questions": {"gap_closed": U.GAP},
                              "targets": {"gap_closed": {"true" if cov else "false": 1.0}},
                              "meta": {"dataset": ds, "kind": "gap_real", "qid": r["id"], "run_gap": float(t["gap_closed"])}})
            st[f"{ds}/{part}: label {int(cov)}"] += 1
            st[f"{ds}/{part}: run closed (>=0.5) while label 0"] += float(t["gap_closed"]) >= 0.5 and not cov
for part in out:
    with open(f"{OUT}/gap_real_{part}.jsonl", "w") as f:
        for x in out[part]: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(json.dumps(dict(st), indent=1))
