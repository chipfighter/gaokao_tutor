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
