#!/usr/bin/env python3
"""Convert a DeFT-RAG run (run_eval.py JSONL) to the GraphRAG-Bench schema, so DeFT-RAG is graded by
the same script (Evaluation/llm_acc_eval.py) and the same request as every baseline.

usage: to_bench.py RUN.jsonl OUT.json
Records that ended in an error get an empty answer (graded incorrect), as in run_eval.py.
"""
import sys, json
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ids = [r["id"] for r in rows]
assert len(ids) == len(set(ids)), "duplicate ids"
out = [{"id": r["id"], "question": r["question"], "question_type": r["hop"],
        "generated_answer": r["answer"] if r.get("error") is None and r["answer"] is not None else "",
        "ground_truth": r["gold"]} for r in rows]
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print(f"{len(out)} rows, {sum(r.get('error') is not None for r in rows)} errors -> {sys.argv[2]}")
