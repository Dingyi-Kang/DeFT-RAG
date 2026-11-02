#!/usr/bin/env python3
"""Build the question files used in the paper from the LogicRAG release: the 1,000 test questions of each benchmark,
in the release's order, with the same ids, question text and gold answers as every system in the paper.

usage: make_questions.py OUT_DIR      (reads $LOGICRAG_DIR/dataset/{musique,2wikimultihopqa,hotpotqa}.json)
writes OUT_DIR/{musique,2wikimultihopqa,hotpotqa}.json: a list of {"id", "question", "gold", "hop"}, where "hop" is
the hop count (MuSiQue: 2hop / 3hop / 4hop) or the question type (2Wiki, HotpotQA); it is used only for reporting.
"""
import os, sys, json, re
LG = os.path.abspath(os.path.expanduser(os.environ.get("LOGICRAG_DIR", "LogicRAG")))
OUT = sys.argv[1] if len(sys.argv) > 1 else "questions"
os.makedirs(OUT, exist_ok=True)
PREFIX = {"musique": "MuSiQue", "2wikimultihopqa": "2Wiki", "hotpotqa": "HotpotQA"}
for d, pre in PREFIX.items():
    rows = json.load(open(os.path.join(LG, "dataset", f"{d}.json")))
    out = []
    for r in rows:
        rid = r.get("id", r.get("_id"))
        hop = re.match(r"(\d)hop", rid).group(0) if d == "musique" else r["type"]
        out.append({"id": f"{pre}-{rid}", "question": r["question"].strip(), "gold": r["answer"], "hop": hop})
    json.dump(out, open(os.path.join(OUT, f"{d}.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{d}: {len(out)} questions -> {OUT}/{d}.json")
