"""Settings: paths and model names, read from the environment with defaults.

Nothing here touches the filesystem at import time. Paths are anchored on the
repo root so the package works from any working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# src/finrag/config.py -> src/finrag -> src -> repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VERTEX_LOCATION = "us-central1"


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


@dataclass
class Settings:
    """Resolved configuration for one process."""

    repo_root: Path = REPO_ROOT
    corpus_dir: Path = REPO_ROOT / "data" / "corpus"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    gold_path: Path = REPO_ROOT / "evals" / "gold" / "gold_set.json"
    results_dir: Path = REPO_ROOT / "evals" / "results"
    policy_path: Path = REPO_ROOT / "policy" / "policy.yaml"
    embed_model: str = DEFAULT_EMBED_MODEL
    vertex_project: str | None = None
    vertex_location: str = DEFAULT_VERTEX_LOCATION
    vertex_model: str = "gemini-2.0-flash"

    @property
    def chunks_path(self) -> Path:
        return self.artifacts_dir / "chunks.jsonl"

    @property
    def bm25_dir(self) -> Path:
        return self.artifacts_dir / "bm25"

    @property
    def dense_path(self) -> Path:
        return self.artifacts_dir / "dense.npz"

    @property
    def graph_path(self) -> Path:
        return self.artifacts_dir / "graph.json"

    @property
    def graph_llm_cache_path(self) -> Path:
        return self.artifacts_dir / "graph_llm_cache.jsonl"

    @classmethod
    def from_env(cls, root: Path | None = None) -> Settings:
        """Build settings from environment variables, falling back to defaults."""
        base = Path(root).expanduser() if root else _path_from_env("FINRAG_ROOT", REPO_ROOT)
        return cls(
            repo_root=base,
            corpus_dir=_path_from_env("FINRAG_CORPUS_DIR", base / "data" / "corpus"),
            artifacts_dir=_path_from_env("FINRAG_ARTIFACTS_DIR", base / "artifacts"),
            gold_path=_path_from_env(
                "FINRAG_GOLD_PATH", base / "evals" / "gold" / "gold_set.json"
            ),
            results_dir=_path_from_env("FINRAG_RESULTS_DIR", base / "evals" / "results"),
            policy_path=_path_from_env("FINRAG_POLICY_PATH", base / "policy" / "policy.yaml"),
            embed_model=os.environ.get("FINRAG_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            vertex_project=os.environ.get("VERTEX_PROJECT") or None,
            vertex_location=os.environ.get("VERTEX_LOCATION") or DEFAULT_VERTEX_LOCATION,
            vertex_model=os.environ.get("VERTEX_MODEL", "gemini-2.0-flash"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, resolved once on first use."""
    return Settings.from_env()
