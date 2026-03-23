"""Unit tests for the RAG engine (loader, indexer, retriever)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag.loader import CHUNK_OVERLAP, CHUNK_SIZE, _guess_year, load_documents


class TestGuessYear:

    def test_extracts_year(self):
        assert _guess_year("math_2024_exam.pdf") == "2024"

    def test_extracts_first_year(self):
        assert _guess_year("2023_2024_exam.pdf") == "2023"

    def test_returns_none_for_no_year(self):
        assert _guess_year("exam.pdf") is None

    def test_ignores_non_2000s(self):
        assert _guess_year("exam_1999.pdf") is None


class TestLoadDocuments:

    def test_loads_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test_2024.txt"
            p.write_text("This is test content for chunking. " * 50, encoding="utf-8")

            docs = load_documents(tmpdir, subject="math", doc_type="exam")

            assert len(docs) >= 1
            assert docs[0].metadata["subject"] == "math"
            assert docs[0].metadata["year"] == "2024"
            assert docs[0].metadata["doc_type"] == "exam"
            assert docs[0].metadata["source_file"] == "test_2024.txt"

    def test_loads_md_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "notes.md"
            p.write_text("# Chapter 1\n\nSome content here.", encoding="utf-8")

            docs = load_documents(tmpdir, subject="chinese")
            assert len(docs) >= 1

    def test_skips_unsupported_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "image.png").write_bytes(b"\x89PNG")
            (Path(tmpdir) / "test.txt").write_text("content", encoding="utf-8")

            docs = load_documents(tmpdir, subject="math")
            assert len(docs) == 1

    def test_skips_empty_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "empty.txt").write_text("", encoding="utf-8")

            docs = load_documents(tmpdir, subject="math")
            assert len(docs) == 0

    def test_raises_on_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/dir", subject="math")

    def test_chunk_params(self):
        assert CHUNK_SIZE == 1000
        assert CHUNK_OVERLAP == 200


class TestIndexer:

    def test_content_id_deterministic(self):
        from src.rag.indexer import _content_id

        doc = Document(page_content="test content", metadata={"source_file": "test.pdf"})
        id1 = _content_id(doc)
        id2 = _content_id(doc)
        assert id1 == id2

    def test_content_id_differs_for_different_content(self):
        from src.rag.indexer import _content_id

        doc1 = Document(page_content="content A", metadata={"source_file": "test.pdf"})
        doc2 = Document(page_content="content B", metadata={"source_file": "test.pdf"})
        assert _content_id(doc1) != _content_id(doc2)

    def test_resolve_persist_dir_absolute(self):
        from src.rag.indexer import _resolve_persist_dir
        import os

        if os.name == "nt":
            result = _resolve_persist_dir("C:\\absolute\\path")
            assert result == "C:\\absolute\\path"
        else:
            result = _resolve_persist_dir("/absolute/path")
            assert result == "/absolute/path"

    def test_resolve_persist_dir_relative(self):
        from src.rag.indexer import _resolve_persist_dir

        result = _resolve_persist_dir("chroma_store/")
        assert Path(result).is_absolute()

    def test_collection_name(self):
        from src.rag.indexer import COLLECTION_NAME
        assert COLLECTION_NAME == "gaokao_docs"


class TestRetriever:

    def test_relevance_threshold_value(self):
        from src.rag.retriever import DEFAULT_TOP_K, RELEVANCE_THRESHOLD
        assert RELEVANCE_THRESHOLD == 0.3
        assert DEFAULT_TOP_K == 5


# ===========================================================================
# TestReranker — SiliconFlow reranker wrapper
# ===========================================================================

class TestReranker:

    @patch("src.rag.reranker.httpx.post")
    def test_returns_reranked_documents(self, mock_post):
        from src.rag.reranker import rerank

        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "results": [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.80},
                ],
            }),
        )

        docs = [
            {"content": "doc A", "source": "a.pdf"},
            {"content": "doc B", "source": "b.pdf"},
        ]
        result = rerank("query", docs, top_n=2)

        assert len(result) == 2
        assert result[0]["content"] == "doc B"
        assert result[0]["rerank_score"] == 0.95
        assert result[1]["content"] == "doc A"

    @patch("src.rag.reranker.httpx.post", side_effect=Exception("network error"))
    def test_graceful_degradation_on_api_failure(self, mock_post):
        from src.rag.reranker import rerank

        docs = [
            {"content": "doc A", "source": "a.pdf"},
            {"content": "doc B", "source": "b.pdf"},
            {"content": "doc C", "source": "c.pdf"},
        ]
        result = rerank("query", docs, top_n=2)

        assert len(result) == 2
        assert result[0]["content"] == "doc A"
        assert result[1]["content"] == "doc B"

    def test_empty_documents(self):
        from src.rag.reranker import rerank

        assert rerank("query", []) == []

    @patch("src.rag.reranker.httpx.post")
    def test_respects_top_n(self, mock_post):
        from src.rag.reranker import rerank

        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                ],
            }),
        )

        docs = [{"content": "doc A", "source": "a.pdf"}, {"content": "doc B", "source": "b.pdf"}]
        result = rerank("query", docs, top_n=1)

        assert len(result) == 1


# ===========================================================================
# TestBM25Search — keyword search via rank-bm25
# ===========================================================================

class TestBM25Search:

    def test_bm25_search_returns_results(self):
        """BM25 search with a pre-built index returns scored docs."""
        import jieba
        import src.rag.retriever as ret

        from rank_bm25 import BM25Okapi
        corpus_texts = ["判别式的计算方法", "二次方程求根公式", "向量叉积"]
        tokenized = [jieba.lcut(t) for t in corpus_texts]
        corpus = [{"content": t, "source": "test.pdf", "metadata": {}} for t in corpus_texts]

        old_idx, old_corpus = ret._bm25_index, ret._bm25_corpus
        try:
            ret._bm25_index = BM25Okapi(tokenized)
            ret._bm25_corpus = corpus
            results = ret._bm25_search("判别式", top_k=2)
            assert len(results) >= 1
            assert results[0]["content"] == "判别式的计算方法"
        finally:
            ret._bm25_index, ret._bm25_corpus = old_idx, old_corpus

    def test_bm25_search_empty_index(self):
        """BM25 search with no index returns empty list."""
        import src.rag.retriever as ret

        old_idx, old_corpus = ret._bm25_index, ret._bm25_corpus
        try:
            ret._bm25_index = None
            ret._bm25_corpus = []
            results = ret._bm25_search("test")
            assert results == []
        finally:
            ret._bm25_index, ret._bm25_corpus = old_idx, old_corpus


# ===========================================================================
# TestMergeAndDedup — merging vector + BM25 results
# ===========================================================================

class TestMergeAndDedup:

    def test_deduplicates_by_content(self):
        from src.rag.retriever import _merge_and_dedup

        vec = [{"content": "same text", "source": "a.pdf", "score": 0.9}]
        bm25 = [{"content": "same text", "source": "a.pdf", "score": 5.0}]

        merged = _merge_and_dedup(vec, bm25)
        assert len(merged) == 1

    def test_merges_unique_docs(self):
        from src.rag.retriever import _merge_and_dedup

        vec = [{"content": "vector doc", "source": "a.pdf", "score": 0.9}]
        bm25 = [{"content": "keyword doc", "source": "b.pdf", "score": 5.0}]

        merged = _merge_and_dedup(vec, bm25)
        assert len(merged) == 2

    def test_vector_results_first(self):
        from src.rag.retriever import _merge_and_dedup

        vec = [{"content": "vec", "source": "a.pdf", "score": 0.9}]
        bm25 = [{"content": "bm25", "source": "b.pdf", "score": 5.0}]

        merged = _merge_and_dedup(vec, bm25)
        assert merged[0]["content"] == "vec"
        assert merged[1]["content"] == "bm25"

    def test_empty_inputs(self):
        from src.rag.retriever import _merge_and_dedup

        assert _merge_and_dedup([], []) == []


# ===========================================================================
# TestHybridRetrieve — full pipeline with mocked dependencies
# ===========================================================================

class TestHybridRetrieve:

    @patch("src.rag.retriever.rerank")
    @patch("src.rag.retriever._bm25_search")
    @patch("src.rag.retriever._get_vectorstore")
    def test_combines_vector_and_bm25(self, mock_vs, mock_bm25, mock_rerank):
        """retrieve() merges vector + BM25 and calls reranker."""
        import src.rag.retriever as ret

        # Mock vector search
        mock_doc = MagicMock()
        mock_doc.page_content = "vector result"
        mock_doc.metadata = {"source_file": "v.pdf"}
        mock_vs.return_value.similarity_search_with_relevance_scores.return_value = [
            (mock_doc, 0.85),
        ]

        # Mock BM25
        mock_bm25.return_value = [
            {"content": "bm25 result", "source": "b.pdf", "score": 3.5, "metadata": {}},
        ]

        # Mock reranker
        mock_rerank.return_value = [
            {"content": "vector result", "source": "v.pdf", "score": 0.85, "rerank_score": 0.95},
            {"content": "bm25 result", "source": "b.pdf", "score": 3.5, "rerank_score": 0.70},
        ]

        result = ret.retrieve("test query", subject="math")

        assert result["is_hit"] is True
        assert len(result["docs"]) == 2
        mock_rerank.assert_called_once()

    @patch("src.rag.retriever.rerank")
    @patch("src.rag.retriever._bm25_search")
    @patch("src.rag.retriever._get_vectorstore")
    def test_returns_results_when_reranker_fails(self, mock_vs, mock_bm25, mock_rerank):
        """If reranker returns original docs (degraded), retrieve still works."""
        import src.rag.retriever as ret

        mock_doc = MagicMock()
        mock_doc.page_content = "doc"
        mock_doc.metadata = {"source_file": "f.pdf"}
        mock_vs.return_value.similarity_search_with_relevance_scores.return_value = [
            (mock_doc, 0.5),
        ]
        mock_bm25.return_value = []

        # Reranker returns original (degraded)
        mock_rerank.return_value = [
            {"content": "doc", "source": "f.pdf", "score": 0.5, "metadata": {"source_file": "f.pdf"}},
        ]

        result = ret.retrieve("test")

        assert len(result["docs"]) == 1
        assert result["is_hit"] is True

    @patch("src.rag.retriever.rerank")
    @patch("src.rag.retriever._bm25_search")
    @patch("src.rag.retriever._get_vectorstore")
    def test_empty_results(self, mock_vs, mock_bm25, mock_rerank):
        """No results from either source → empty docs, is_hit=False."""
        import src.rag.retriever as ret

        mock_vs.return_value.similarity_search_with_relevance_scores.return_value = []
        mock_bm25.return_value = []

        result = ret.retrieve("nothing")

        assert result["docs"] == []
        assert result["is_hit"] is False
        mock_rerank.assert_not_called()


# ===========================================================================
# TestSettingsIntegration — rag config values
# ===========================================================================

class TestRagSettings:

    def test_rag_vector_top_k(self):
        from src.config import get_setting
        assert get_setting("rag.vector_top_k") == 10

    def test_rag_bm25_top_k(self):
        from src.config import get_setting
        assert get_setting("rag.bm25_top_k") == 10

    def test_rag_reranker_top_n(self):
        from src.config import get_setting
        assert get_setting("rag.reranker_top_n") == 5

    def test_rag_relevance_threshold(self):
        from src.config import get_setting
        assert get_setting("rag.relevance_threshold") == 0.3

    def test_rag_reranker_model(self):
        from src.config import get_setting
        assert get_setting("rag.reranker_model") == "BAAI/bge-reranker-v2-m3"
