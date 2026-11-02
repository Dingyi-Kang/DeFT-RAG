#!/usr/bin/env python3
"""Option C training files from the full build (build_ft_data.py output), without changing any request:
  questions: keep a question (all its kinds) if a seeded hash of its id falls in the first KEEP share (~1,500 of
             3,000 per dataset). The train / dev assignment of build_ft_data.py is kept as it is.
  relevance rows: every request still shows ALL its passages (realistic context); targets are kept for every gold
             passage and for the NEG highest-ranked non-gold passages (p1 is the top retrieval hit for that
             request's own question -- main question or sub-question), i.e. the hardest distractors. Other
             relevance questions get no target, so the trainer skips them. sufficient / depth / gap targets: all kept.
Writes OUT/train.jsonl, OUT/dev_monitor.jsonl (dev, same selection: loss/accuracy during training) and
OUT/dev_full.jsonl (dev, ALL targets: realistic yes/no ratios, for calibration and threshold fitting later).
10-passage window: every relevance request the model sees has 10 passages, so the 20-passage
"rel20" requests of the build are dropped (DeFT-RAG with StartLux scores 20 passages as two windows of 10).
NEGATIVES ARE DRAWN AT RANDOM (2026-10-09, v2): v1 kept the NEG highest-ranked non-gold passages; the model then
learned that a labelled passage further down the list is gold (a position shortcut: non-gold passages at ranks 4-10
got p(yes) 0.30 vs 0.12 at ranks 1-3, relevance AUC 0.919 vs the base 0.932). Now NEG non-gold passages are drawn
uniformly from the request's own top 10 (seeded per request), so a labelled passage's position carries only its
natural information. --hard restores v1.
usage: select_rows.py IN_DIR OUT_DIR [KEEP=0.5] [NEG=3] [--hard]"""
import os, sys, json, hashlib, collections, random
HARD = "--hard" in sys.argv; sys.argv = [x for x in sys.argv if x != "--hard"]
IN, OUT = sys.argv[1], sys.argv[2]; KEEP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5; NEG = int(sys.argv[4]) if len(sys.argv) > 4 else 3
os.makedirs(OUT, exist_ok=True)
keep_q = lambda ds, qid: int(hashlib.md5(f"select:{ds}:{qid}".encode()).hexdigest(), 16) % 1000 < KEEP * 1000
def select(r):
    rel = sorted((k for k in r["targets"] if k.startswith("rel")), key=lambda k: int(k[3:]))   # rank order p1, p2, ...
    gold = [k for k in rel if "true" in r["targets"][k]]
    non = [k for k in rel if "true" not in r["targets"][k]]
    if HARD: hard = non[:NEG]
    else: hard = random.Random(f"neg:{r['meta']['dataset']}:{r['meta']['kind']}:{r['meta']['qid']}").sample(non, min(NEG, len(non)))
    t = {k: v for k, v in r["targets"].items() if not k.startswith("rel") or k in gold or k in hard}
    return dict(r, targets=t)
st = collections.Counter()
for part, outs in (("train", [("train.jsonl", True)]), ("dev", [("dev_monitor.jsonl", True), ("dev_full.jsonl", False)])):
    rows = [json.loads(l) for l in open(f"{IN}/{part}.jsonl")]
    for name, sel in outs:
        with open(f"{OUT}/{name}", "w") as f:
            for r in rows:
                if not keep_q(r["meta"]["dataset"], r["meta"]["qid"]): continue
                if r["meta"]["kind"] == "rel20": st[f"{name}: rel20 dropped (10-passage window)"] += 1; continue
                x = select(r) if sel else r
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
                st[f"{name}: requests"] += 1; st[f"{name}: rows (targets)"] += len(x["targets"])
                st[f"{name}: rows yes"] += sum("true" in v for k, v in x["targets"].items())
                st[f"{name}: {x['meta']['dataset']}/{x['meta']['kind']}"] += 1
json.dump(dict(st), open(f"{OUT}/select_report.json", "w"), indent=1); print(json.dumps(dict(st), indent=1))
