"""Dense retriever of DeFT-RAG: Qwen3-Embedding-4B over a benchmark corpus, exactly as used for the paper's results.

Passages are encoded as "Title: {title}. Content: {text}" (the format of the LogicRAG release, whose corpora we use).
The embedder is a decoder model, so it uses last-token pooling with left padding; queries (not passages) carry the
model's instruction prefix; texts are cut at 512 tokens, encoded in batches of 8, and L2-normalised, so the dot
product is the cosine similarity. Corpus embeddings are computed once and cached on disk.
"""
import os, json, hashlib
import torch
from transformers import AutoTokenizer, AutoModel

EMBEDDING_MODEL = os.environ.get("DEFTRAG_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
QUERY_INSTRUCTION = "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: "
MAX_LENGTH, BATCH_SIZE = 512, 8
GPU_RESERVE_GIB = float(os.environ.get("GPU_RESERVE_GIB", "1.5"))


def _max_memory():
    """Per-GPU budget for device_map="auto", from FREE memory, keeping GPU_RESERVE_GIB free on every card."""
    caps = {i: f"{max(torch.cuda.mem_get_info(i)[0] / 1024 ** 3 - GPU_RESERVE_GIB, 0):.1f}GiB"
            for i in range(torch.cuda.device_count())}
    caps["cpu"] = "16GiB"
    return caps


class Embedder:
    def __init__(self, name=EMBEDDING_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
        if torch.cuda.is_available():
            self.model = AutoModel.from_pretrained(name, device_map="auto", max_memory=_max_memory()).eval()
            self.device = next(self.model.parameters()).device
        else:
            self.model, self.device = AutoModel.from_pretrained(name).eval(), "cpu"

    @torch.no_grad()
    def encode(self, texts, is_query=False):
        """L2-normalised embeddings (float32, CPU) of a list of texts."""
        if is_query:
            texts = [QUERY_INSTRUCTION + t for t in texts]
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            enc = self.tokenizer(texts[i:i + BATCH_SIZE], padding=True, truncation=True, max_length=MAX_LENGTH,
                                 return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"]
            if mask[:, -1].sum() == mask.shape[0]:          # left padding: the last position is a real token
                vec = hidden[:, -1]
            else:
                vec = hidden[torch.arange(hidden.shape[0], device=hidden.device), mask.sum(dim=1) - 1]
            out.append(torch.nn.functional.normalize(vec.float(), p=2, dim=1).cpu())
        return torch.cat(out)


class Retriever:
    """corpus_list: the corpus records; corpus_embeddings: one row per passage; encode_query(q) -> vector."""

    def __init__(self, corpus_path, cache_dir=None):
        self.corpus_list = json.load(open(corpus_path))
        self.embedder = Embedder()
        cache_dir = cache_dir or os.environ.get("DEFTRAG_EMB_CACHE", "emb_cache")
        os.makedirs(cache_dir, exist_ok=True)
        key = hashlib.sha1(open(corpus_path, "rb").read()).hexdigest()[:12]
        cache = os.path.join(cache_dir, f"{os.path.basename(corpus_path)[:-5]}_{key}.pt")
        if os.path.exists(cache):
            self.corpus_embeddings = torch.load(cache)
        else:
            print(f"encoding {len(self.corpus_list)} passages of {corpus_path} (once; cached in {cache})", flush=True)
            self.corpus_embeddings = self.embedder.encode(
                [f"Title: {d['title']}. Content: {d['text']}" for d in self.corpus_list])
            torch.save(self.corpus_embeddings, cache)

    def encode_query(self, query):
        return self.embedder.encode([query], is_query=True)[0]
