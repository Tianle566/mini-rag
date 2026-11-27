"""mini-rag —— 无依赖的最小 RAG 实现。

设计取舍（也是这个 demo 想讲清楚的事）：

  1. **不需要向量库**。检索的本质是「按相关度排序」，BM25 是最经典的词频方案，
     纯标准库 40 行就能写完。embedding 解决的是「同义词/语义相近」，不是「检索」。
  2. **中文不装分词器**。用字符二元组（bigram）代替分词：中文没有空格，
     但相邻两个字组成的片段已经足够区分主题，且零依赖。
  3. **生成必须带引用**。提示词强制模型只依据资料回答，并标出 [1][2]；
     没有资料就说不知道——这是 RAG 相对纯 LLM 的唯一硬价值。

用法：
    python rag.py index                      # 建索引并打印统计
    python rag.py ask "为什么不能把 tok/latency 当吞吐"
    python rag.py ask "..." --k 5 --show-chunks
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.request

BASE_URL = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:7863")
MODEL = os.environ.get("RAG_MODEL", "deepseek-v4.1-flash")
CONFIG = os.environ.get("RAG_CONFIG", r"E:\WorkBuddyManager\upstream\config.json")

CHUNK_SIZE = 360      # 目标字符数（中文按字算）
CHUNK_OVERLAP = 80    # 相邻块重叠，避免答案被切断
K1, B = 1.5, 0.75     # BM25 参数


# ---------------------------------------------------------------- 语料与切块

def load_docs(corpus_dir: str) -> list[dict]:
    docs = []
    for name in sorted(os.listdir(corpus_dir)):
        if not name.endswith((".md", ".txt")):
            continue
        path = os.path.join(corpus_dir, name)
        with open(path, encoding="utf-8") as fh:
            docs.append({"doc": name, "text": fh.read()})
    return docs


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按段落累积切块，超长再滑动切。返回非空块列表。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}" if buf else p
            continue
        if buf:
            chunks.append(buf)
        if len(p) <= size:
            buf = p
            continue
        step = max(1, size - overlap)
        for i in range(0, len(p), step):
            piece = p[i:i + size].strip()
            if piece:
                chunks.append(piece)
        buf = ""
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def tokenize(text: str) -> list[str]:
    """英文/数字按词，中文按字符二元组。"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_./-]+", text)
    han = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend("".join(pair) for pair in zip(han, han[1:]))
    return tokens


# ---------------------------------------------------------------- BM25

class BM25:
    def __init__(self, docs: list[str]):
        self.docs = docs
        self.toks = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.toks]
        self.avg = sum(self.doc_len) / max(1, len(self.doc_len))
        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for toks in self.toks:
            counter: dict[str, int] = {}
            for t in toks:
                counter[t] = counter.get(t, 0) + 1
            self.tf.append(counter)
            for t in counter:
                df[t] = df.get(t, 0) + 1
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query: str, idx: int) -> float:
        q = tokenize(query)
        tf = self.tf[idx]
        dl = self.doc_len[idx] or 1
        total = 0.0
        for t in set(q):
            if t not in self.idf:
                continue
            f = tf.get(t, 0)
            if not f:
                continue
            denom = f + K1 * (1 - B + B * dl / (self.avg or 1))
            total += self.idf[t] * f * (K1 + 1) / denom
        return total

    def search(self, query: str, k: int = 4) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# ---------------------------------------------------------------- 生成

def build_messages(question: str, hits: list[str]) -> list[dict]:
    ctx = "\n\n".join(f"[{i + 1}] {h}" for i, h in enumerate(hits))
    return [
        {"role": "system", "content": (
            "你是一个只依据资料回答的助手。规则：\n"
            "1) 只能使用下面给出的资料，不要补充资料以外的知识；\n"
            "2) 每个结论后面标注来源编号，如 [1]；\n"
            "3) 资料没有提到的，直接回答「资料中没有相关内容」，不要猜。"
        )},
        {"role": "user", "content": f"资料：\n{ctx}\n\n问题：{question}"},
    ]


def api_key() -> str:
    with open(CONFIG, encoding="utf-8") as fh:
        return str(json.load(fh).get("api_key") or "")


def chat(messages: list[dict], temperature: float = 0) -> str:
    body = json.dumps({"model": MODEL, "messages": messages,
                       "temperature": temperature, "stream": False}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key()}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------- CLI

def build_index(corpus_dir: str):
    docs = load_docs(corpus_dir)
    chunks, meta = [], []
    for d in docs:
        for c in chunk_text(d["text"]):
            chunks.append(c)
            meta.append(d["doc"])
    return docs, chunks, meta, BM25(chunks)


def main() -> None:
    ap = argparse.ArgumentParser(description="mini-rag：无依赖的检索增强生成 demo")
    ap.add_argument("cmd", choices=["index", "ask"])
    ap.add_argument("question", nargs="?", default="")
    ap.add_argument("--corpus", default=os.path.join(os.path.dirname(__file__), "corpus"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--show-chunks", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="只检索，不调用模型")
    args = ap.parse_args()

    docs, chunks, meta, index = build_index(args.corpus)
    if args.cmd == "index":
        print(f"文档 {len(docs)} 篇 → 切块 {len(chunks)} 个")
        for d in docs:
            n = sum(1 for m in meta if m == d["doc"])
            print(f"  {d['doc']:<44s} {len(d['text']):>6d} 字  →  {n} 块")
        return

    if not args.question:
        print("请给出问题，例如：python rag.py ask \"为什么不能用 tok/latency\"")
        sys.exit(2)

    hits = index.search(args.question, k=args.k)
    if not hits:
        print("没有检索到相关内容。")
        sys.exit(0)

    print(f"问题：{args.question}\n")
    print("检索结果：")
    picked = []
    for rank, (i, sc) in enumerate(hits, 1):
        picked.append(chunks[i])
        head = chunks[i].replace("\n", " ")[:70]
        print(f"  [{rank}] score={sc:6.2f}  {meta[i]:<40s} {head}…")
        if args.show_chunks:
            print("      " + chunks[i].replace("\n", " ")[:200] + "…")
    print()
    if args.no_llm:
        print("（--no-llm：跳过生成）")
        return
    answer = chat(build_messages(args.question, picked))
    print("回答：")
    print(answer.strip())


if __name__ == "__main__":
    main()
