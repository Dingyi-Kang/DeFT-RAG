#!/usr/bin/env python3
"""Run DeFT-RAG on a benchmark question set.

usage: run_eval.py DATASET QUESTIONS.json OUT.jsonl

  DATASET        musique | 2wikimultihopqa | hotpotqa  (selects the corpus in $LOGICRAG_DIR/dataset/)
  QUESTIONS.json a JSON list of {"id", "question", "gold", "hop"}; "hop" is the hop count or question type
                 (only used for reporting)
  OUT.jsonl      one JSON line per finished question (answer, trace, counters); finished ids are skipped,
                 so an interrupted run resumes with the same command

Options (environment):
  DEFTRAG_IDS=file      run only the question ids listed in the file (one per line)
  DEFTRAG_FIXES=...     which rules are on (see loop.py; unset = the accepted set used in the paper)
  DEFTRAG_REUSE=1       turn on the QA cache (needs DEFTRAG_CACHE_DIR); default off
  DEFTRAG_LLM, DEFTRAG_LLM_URL, DEFTRAG_LLM_KEY      the backbone LLM (see loop.py)
  DEFTRAG_DECIDER_URL, DEFTRAG_DECIDER_FORMAT=v2     the fine-tuned decision model (see README)

Grading: convert OUT.jsonl with to_bench.py and grade it with GraphRAG-Benchmark's Evaluation/llm_acc_eval.py,
as for every baseline. Stops after MAX_CONSEC_ERR consecutive errors, so an API outage cannot write many
empty records; rerun the same command to resume.
"""
import os, sys, json, time, hashlib, traceback
if len(sys.argv) != 4: raise SystemExit(__doc__)
DATASET, QFILE, OUT = sys.argv[1], os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
os.environ["DEFTRAG_DATASET"] = DATASET
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MAX_CONSEC_ERR = 5
CODE = hashlib.sha1(open(f"{HERE}/loop.py", "rb").read()).hexdigest()[:12]

Q = [{k: r[k] for k in ("id", "hop", "question", "gold")} for r in json.load(open(QFILE))]
if os.environ.get("DEFTRAG_IDS"):
    keep = {l.strip() for l in open(os.path.abspath(os.environ["DEFTRAG_IDS"])) if l.strip()}
    Q = [q for q in Q if q["id"] in keep]

import loop as U

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        if l.strip():
            r = json.loads(l)
            if r.get("code") != CODE or r.get("llm") != U.LLM_MODEL or r.get("fixes") != sorted(U.FIXES):
                raise SystemExit(f"{OUT} holds results from another code version, LLM or rule set; use a new file")
            done.add(r["id"])
print(f"dataset={DATASET} n={len(Q)} done={len(done)} code={CODE} fixes={sorted(U.FIXES)} reuse={U.REUSE} "
      f"llm={U.LLM_MODEL}", flush=True)

consec = 0
for i, q in enumerate(Q, 1):
    if q["id"] in done: continue
    t0 = time.time(); c0 = dict(U.COUNTERS)
    rec = dict(q, code=CODE, version=U.VERSION, dataset=U.DATASET, llm=U.LLM_MODEL, fixes=sorted(U.FIXES),
               reuse=U.REUSE)
    try:
        ans, trace = U.solve(q["question"], qid=q["id"])
        rec["answer_secs"] = round(time.time() - t0, 2)
        steps = [t for t in trace if "step" in t and t["step"] > 0]
        rec.update({"answer": ans, "exit": next((t["exit"] for t in reversed(trace) if "exit" in t), None),
                    "n_steps": len(steps), "fallback": bool(trace[-1].get("fallback")), "trace": trace, "error": None})
        consec = 0
    except Exception as e:
        consec += 1
        rec.update({"answer": None, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]})
    rec["counters"] = {k: round(U.COUNTERS[k] - c0.get(k, 0), 3) for k in U.COUNTERS}
    with open(OUT, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{i}/{len(Q)}] exit={rec.get('exit')} steps={rec.get('n_steps')} {time.time() - t0:.1f}s | "
          f"gold={q['gold'][:40]!r} ans={str(rec['answer'])[:40]!r}", flush=True)
    if consec >= MAX_CONSEC_ERR:
        print(f"ABORT: {consec} consecutive errors, last: {rec['error']}", flush=True); sys.exit(2)
print("DONE", flush=True)
