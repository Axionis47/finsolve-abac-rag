"""The chunker registry, and the settings that pick a namespace by it."""

from __future__ import annotations

from pathlib import Path

import pytest

from finrag.config import Settings
from finrag.ingest.chunker import chunk_markdown
from finrag.ingest.registry import available_chunkers, build_chunker


def test_available_chunkers_contains_heading() -> None:
    assert "heading" in available_chunkers()


def test_available_chunkers_is_sorted() -> None:
    assert available_chunkers() == sorted(available_chunkers())


def test_build_chunker_heading_is_chunk_markdown() -> None:
    assert build_chunker("heading") is chunk_markdown


def test_unknown_chunker_names_the_known_ones() -> None:
    with pytest.raises(KeyError) as excinfo:
        build_chunker("nope")
    message = str(excinfo.value)
    assert "nope" in message
    assert "heading" in message
    assert "table_aware" in message


def test_available_chunkers_contains_table_aware() -> None:
    assert "table_aware" in available_chunkers()


def test_default_settings_keep_todays_flat_layout(repo_root: Path) -> None:
    settings = Settings.from_env(root=repo_root)
    artifacts = settings.artifacts_dir
    assert settings.chunker == "heading"
    assert settings.chunks_path == artifacts / "chunks.jsonl"
    assert settings.bm25_dir == artifacts / "bm25"
    assert settings.dense_path == artifacts / "dense.npz"
    assert settings.embed_cache_path == artifacts / "embed_cache.npz"


def test_non_default_chunker_namespaces_index_paths(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINRAG_CHUNKER", "table_aware")
    settings = Settings.from_env(root=repo_root)
    artifacts = settings.artifacts_dir

    assert settings.chunker == "table_aware"
    assert settings.chunks_path == artifacts / "table_aware" / "chunks.jsonl"
    assert settings.bm25_dir == artifacts / "table_aware" / "bm25"
    assert settings.dense_path == artifacts / "table_aware" / "dense.npz"


def test_non_default_chunker_leaves_shared_artifacts_at_the_root(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The embed cache is keyed by text hash and model, so every chunker shares
    it; the graph is built from the default chunker's chunks only. Neither
    moves when FINRAG_CHUNKER changes.
    """
    monkeypatch.setenv("FINRAG_CHUNKER", "table_aware")
    settings = Settings.from_env(root=repo_root)
    artifacts = settings.artifacts_dir

    assert settings.embed_cache_path == artifacts / "embed_cache.npz"
    assert settings.graph_path == artifacts / "graph.json"
    assert settings.graph_llm_cache_path == artifacts / "graph_llm_cache.jsonl"
