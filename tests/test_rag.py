"""mini-rag 的测试：只测真正决定结论的部分。

不测网络、不测模型——那属于集成测试，会随上游抖动而误报。
这里钉住四件事：切块、分词、排序、提示词。
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag  # noqa: E402


class ChunkTest(unittest.TestCase):
    def test_no_empty_chunks(self):
        text = "第一段。\n\n\n\n第二段。\n\n"
        chunks = rag.chunk_text(text)
        self.assertTrue(chunks)
        self.assertTrue(all(c.strip() for c in chunks))

    def test_respects_size(self):
        text = "\n\n".join("第%d段，" % i + "内容" * 50 for i in range(20))
        chunks = rag.chunk_text(text, size=200, overlap=40)
        # 允许重叠导致的少量超出，但不该出现成倍膨胀
        self.assertTrue(all(len(c) <= 260 for c in chunks), max(len(c) for c in chunks))

    def test_long_paragraph_split_with_overlap(self):
        text = "甲" * 1000
        chunks = rag.chunk_text(text, size=300, overlap=60)
        self.assertGreaterEqual(len(chunks), 3)
        # 相邻块首尾应有重叠
        self.assertTrue(chunks[0][-20:] in chunks[1] or chunks[1][:20] in chunks[0])


class TokenizeTest(unittest.TestCase):
    def test_chinese_bigram(self):
        toks = rag.tokenize("吞吐量")
        self.assertIn("吞吐", toks)
        self.assertIn("吐量", toks)

    def test_english_and_numbers(self):
        toks = rag.tokenize("HTTP 429 Error")
        self.assertIn("http", toks)
        self.assertIn("429", toks)


class RankingTest(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "BM25 是词频检索算法，用逆文档频率给稀有词加权。",
            "分块太大会降低检索精度，太小会切断语义。",
            "向量检索能处理同义词，但对专有名词不如词频准。",
        ]
        self.index = rag.BM25(self.docs)

    def test_hits_expected_doc(self):
        hits = self.index.search("分块大小怎么选", k=1)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], 1)

    def test_irrelevant_query_returns_nothing(self):
        self.assertEqual(self.index.search("量子纠缠"), [])

    def test_score_monotonic(self):
        hits = self.index.search("词频 检索", k=3)
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))


class PromptTest(unittest.TestCase):
    def test_citations_and_rule(self):
        msgs = rag.build_messages("问题？", ["资料甲", "资料乙"])
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("只能使用下面给出的资料", msgs[0]["content"])
        self.assertIn("[1] 资料甲", msgs[1]["content"])
        self.assertIn("[2] 资料乙", msgs[1]["content"])

    def test_no_context_no_crash(self):
        msgs = rag.build_messages("问题？", [])
        self.assertIn("问题？", msgs[1]["content"])


class PipelineTest(unittest.TestCase):
    """端到端：不联网，模型用假的。"""

    def test_ask_pipeline(self):
        with mock.patch.object(rag, "chat", return_value="答案 [1]") as fake:
            docs, chunks, meta, index = rag.build_index(
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus"))
            self.assertGreaterEqual(len(docs), 3)
            hits = index.search("为什么不能用 tok/latency 当吞吐", k=3)
            self.assertTrue(hits)
            answer = rag.chat(rag.build_messages("为什么不能用 tok/latency 当吞吐",
                                                 [chunks[i] for i, _ in hits]))
            self.assertIn("[1]", answer)
            fake.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
