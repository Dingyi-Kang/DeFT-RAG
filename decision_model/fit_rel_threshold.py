#!/usr/bin/env python3
"""Fit DeFT-RAG's yes/no relevance cut-off on TRAINING-split dev data (score_dev_relevance.py output): the highest
cut-off whose pooled gold recall is >= TARGET (default 0.95).
Probabilities are first moved to the model's FINAL (calibrated) temperature, exactly: logit' = logit * T_scored / T_final.
Reports recall / precision / passages kept per request at the chosen cut-off, pooled and per dataset and kind.
usage: fit_rel_threshold.py SCORES.jsonl FINAL_CONFIG.json [TARGET=0.95]"""
import sys, json, math, collections
S, CFG = sys.argv[1], sys.argv[2]; TARGET = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95
rows = [json.loads(l) for l in open(S)]; t0 = rows[0]["temps"]["noul"]; rows = rows[1:]
t1 = json.load(open(CFG))["temperature_by_type"]["noul"]
def tr(p):
    p = min(max(p, 1e-9), 1 - 1e-9); z = math.log(p / (1 - p)) * t0 / t1; return 1 / (1 + math.exp(-z))
P = [(tr(p), y, r["dataset"], r["kind"]) for r in rows for p, y in zip(r["p"], r["y"])]
ngold = sum(y for _, y, _, _ in P)
def stats(tau, sub):
    g = sum(y for _, y, *_ in sub); k = sum(p >= tau for p, *_ in sub); kg = sum(p >= tau and y for p, y, *_ in sub)
    return kg / max(1, g), kg / max(1, k), k
cands = sorted({p for p, *_ in P}, reverse=True)
tau = next(t for t in cands if stats(t, P)[0] >= TARGET)
nreq = len(rows)
print(f"temperatures: scored {t0}, final {t1}; dev requests {nreq}, relevance rows {len(P)}, gold {ngold}")
for t in (0.5, tau):
    r, pr, k = stats(t, P); print(f"cut-off {t:.4f}: recall {r:.3f}, precision {pr:.3f}, kept {k / nreq:.2f} per request")
for key in sorted({(d, kd) for _, _, d, kd in P}):
    sub = [x for x in P if (x[2], x[3]) == key]; r, pr, k = stats(tau, sub)
    print(f"   {key[0]}/{key[1]}: recall {r:.3f}, precision {pr:.3f}")
print(f"FITTED_TAU={tau:.4f}")
