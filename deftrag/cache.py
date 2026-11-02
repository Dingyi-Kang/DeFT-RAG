#!/usr/bin/env python3
"""QA cache of DeFT-RAG: answers of self-contained sub-questions, reused across questions.

Rules:
  C1  one entry per normalised sub-query, held by its FIRST occurrence (answered or failed)
  C2  lookups ignore entries created by the current question (reuse is across questions)
  C3  exact tier: any entry with the same key -- a cached answer is reused as-is, a cached
      failure is returned as a failure
  C4  similarity tier: only ANSWERED entries; the k nearest by the sub-query's retrieval
      embedding (cosine); the caller asks the DM whether one of them is the same question
  C6  the cache lives on disk (entries.jsonl, one line per entry, embedding included), so a
      resumed run continues with the same cache; a folder is tied to one configuration via
      manifest.json and refuses to open under a different one
"""
import os, json
import torch


class SubQueryCache:
    def __init__(self, path, manifest):
        os.makedirs(path, exist_ok=True)
        self.path = path
        mf = os.path.join(path, "manifest.json")
        if os.path.exists(mf):
            old = json.load(open(mf))
            if old != manifest:
                raise SystemExit(f"cache at {path} was built under a different configuration; "
                                 f"use a new DEFTRAG_CACHE_DIR")
        else:
            with open(mf, "w") as f: json.dump(manifest, f, indent=1)
        self.entries, self.by_key, self._emb = [], {}, None
        ep = os.path.join(path, "entries.jsonl")
        if os.path.exists(ep):
            for line in open(ep):
                if line.strip(): self._add(json.loads(line))

    # ---- internal ----
    def _add(self, e):
        self.entries.append(e); self.by_key[e["key"]] = e
        v = torch.tensor(e["emb"], dtype=torch.float).unsqueeze(0)
        self._emb = v if self._emb is None else torch.cat([self._emb, v])

    # ---- C3: exact ----
    def exact(self, key, qid):
        e = self.by_key.get(key)
        return e if e is not None and e["qid"] != qid else None

    # ---- C4: similarity candidates ----
    def candidates(self, emb, qid, k=5):
        """k nearest ANSWERED entries from other questions, nearest first, as (entry, cosine)."""
        if self._emb is None: return []
        idx = [i for i, e in enumerate(self.entries) if e["status"] == "answer" and e["qid"] != qid]
        if not idx: return []
        q = torch.nn.functional.normalize(emb.float().reshape(1, -1), dim=1)
        M = torch.nn.functional.normalize(self._emb[idx], dim=1)
        sims = (M @ q.T).squeeze(1)
        order = torch.argsort(sims, descending=True)[:k].tolist()
        return [(self.entries[idx[j]], float(sims[j])) for j in order]

    # ---- C1: insert the first occurrence only ----
    def insert(self, key, sq, status, answer, ids, qid, step, emb):
        if key in self.by_key: return False
        e = {"key": key, "sq": sq, "status": status, "answer": answer, "ids": list(ids),
             "qid": qid, "step": step, "emb": [round(float(x), 6) for x in emb.reshape(-1).tolist()]}
        with open(os.path.join(self.path, "entries.jsonl"), "a") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self._add(e)
        return True
