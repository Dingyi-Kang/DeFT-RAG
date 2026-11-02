#!/usr/bin/env python3
"""LLM-judged accuracy (Acc) of a run, exactly as graded in the paper.

usage: grade.py --judge MODEL --url URL [--key KEY] [--api ollama|openai] BENCH.json OUT.json

  BENCH.json  a run converted with to_bench.py ({"id", "question", "question_type", "generated_answer", "ground_truth"})
  --api       ollama: the native Ollama /api/chat endpoint (URL e.g. http://localhost:11434), reasoning turned off;
              openai: any OpenAI-compatible chat-completions endpoint (URL e.g. https://<endpoint>/v1)
  As in the paper, a run is never graded by its own backbone: gpt-4o-mini runs were judged by Qwen3.6-35B-A3B
  (qwen3.6:35b-a3b-q8_0 on Ollama) and Qwen3.6 runs by gpt-4o-mini.

The judge sees the question, the gold answer and the predicted answer with SimpleQA's grading criteria (Wei et al.,
2024), reduced to a binary verdict, in one user message; temperature 0, top-p 1, seed 42, at most 16 output tokens.
A question whose judge request fails is left out and reported. OUT.json holds per-question verdicts and the averages
per question type and overall.
"""
import argparse, json, sys, time, urllib.request, urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

SEED = 42
SYSTEM = 'You are an expert evaluator.\n        '
USER = "Your job is to look at a question, a gold target, and a predicted answer, and grade the predicted answer as either CORRECT or INCORRECT.\n\n        Question: {question}\n        Gold target: {gold_ans}\n        Predicted answer: {pre_answer}\n\n        Grade CORRECT if the predicted answer fully contains the important information in\n        the gold target and does not contradict it. Only semantic meaning matters;\n        capitalization, punctuation, grammar, and order do not matter. Hedging is acceptable\n        provided the important information is present.\n\n        Grade INCORRECT if any factual statement in the predicted answer contradicts the gold\n        target, or if the important information in the gold target is absent.\n\n        Respond with ONLY 'correct' or 'incorrect'.\n        Response:\n        "


def prompt(item):
    return f"{SYSTEM}\n\n" + USER.format(question=item.get("question", ""), gold_ans=item.get("ground_truth", ""),
                                          pre_answer=item.get("generated_answer", ""))


def _post(url, body, key):
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
            return json.load(urllib.request.urlopen(req, timeout=300))
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 7: raise
            time.sleep(min(60, 2 * 2 ** attempt))


def judge(item, a):
    p = prompt(item)
    if a.api == "ollama":
        r = _post(a.url.rstrip("/") + "/api/chat",
                  {"model": a.judge, "messages": [{"role": "user", "content": p}], "stream": False, "think": False,
                   "options": {"temperature": 0.0, "num_ctx": 32768, "num_predict": 16, "top_p": 1, "seed": SEED}}, a.key)
        text = r["message"]["content"]
    else:
        r = _post(a.url.rstrip("/") + "/chat/completions",
                  {"model": a.judge, "messages": [{"role": "user", "content": p}], "temperature": 0.0, "top_p": 1,
                   "seed": SEED, "max_tokens": 16}, a.key)
        text = r["choices"][0]["message"]["content"] or ""
    text = text.strip().lower()
    return 0.0 if "incorrect" in text else (1.0 if "correct" in text else 0.0)   # 'incorrect' contains 'correct'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench"); ap.add_argument("out")
    ap.add_argument("--judge", required=True); ap.add_argument("--url", required=True); ap.add_argument("--key", default="")
    ap.add_argument("--api", choices=["ollama", "openai"], default="ollama"); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    data = json.load(open(a.bench))

    def one(item):
        try: return {"id": item.get("id"), "question_type": item.get("question_type"), "llm_acc": judge(item, a)}
        except Exception as e:
            print(f"judge failed for {item.get('id')}: {e}", flush=True); return None

    with ThreadPoolExecutor(a.workers) as ex: rows = [r for r in ex.map(one, data) if r]
    by_type = defaultdict(list)
    for r in rows: by_type[r["question_type"] or "all"].append(r["llm_acc"])
    out = {"accuracy": sum(r["llm_acc"] for r in rows) / len(rows), "n_graded": len(rows), "n_failed": len(data) - len(rows),
           "by_type": {k: {"accuracy": sum(v) / len(v), "n": len(v)} for k, v in sorted(by_type.items())}, "detailed": rows}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"accuracy {out['accuracy']:.3f} on {len(rows)} graded questions ({out['n_failed']} failed) -> {a.out}")


if __name__ == "__main__":
    main()
