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

# The Vertex text embedding model dense retrieval uses, and its vector length.
DEFAULT_VERTEX_EMBED_MODEL = "text-embedding-005"

# The Gemini model the answer step generates with, and the fallback for anything
# else that asks for a model by no other name. It was gemini-2.0-flash from R0
# until R6; Vertex now answers 404 NOT_FOUND for that name in this project, and a
# default that cannot be reached is a trap for whoever clones this next.
# gemini-2.5-flash works in us-central1. gemini-3.6-flash works too, but only on
# VERTEX_LOCATION=global, so it is not the default while .env pins a region.
DEFAULT_VERTEX_MODEL = "gemini-2.5-flash"

# The Gemini model the independent LLM reranker calls. Read from our own variable, not
# the `.env` VERTEX_MODEL, for the same reason as the embedding model: a copied
# `.env` must not silently change the model a published number was measured with.
# gemini-2.0-flash is not served to this project (404); 2.5-flash is.
DEFAULT_VERTEX_RERANK_MODEL = "gemini-2.5-flash"

# Which embedding backend `finrag ingest` and dense search should use. "vertex"
# calls the real API; "fake" is the offline deterministic stand-in the tests use.
DEFAULT_EMBEDDER = "vertex"

# Which chunking strategy `finrag ingest` should use. The literal is repeated
# here rather than imported, to keep this widely-imported module free of a
# dependency on finrag.ingest; see finrag.ingest.registry.DEFAULT_CHUNKER for
# the strategy this name resolves to.
DEFAULT_CHUNKER = "heading"


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def load_env(path: str | Path | None = None) -> list[str]:
    """Read `KEY=VALUE` lines from a `.env` file into `os.environ`.

    A value already in the environment always wins, so the shell can override
    the file and a test's monkeypatched variable is never clobbered. Returns the
    names it set, never the values, because this file holds credentials.

    Deliberately not python-dotenv: our `.env` is plain `KEY=VALUE` lines with
    `#` comments, and one dependency for six lines of parsing is not worth it.
    """
    env_path = Path(path) if path else REPO_ROOT / ".env"
    if not env_path.exists():
        return []

    added: list[str] = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name and name not in os.environ:
            os.environ[name] = value.strip().strip('"').strip("'")
            added.append(name)
    return added


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
    vertex_model: str = DEFAULT_VERTEX_MODEL
    vertex_embed_model: str = DEFAULT_VERTEX_EMBED_MODEL
    vertex_rerank_model: str = DEFAULT_VERTEX_RERANK_MODEL
    embedder: str = DEFAULT_EMBEDDER
    chunker: str = DEFAULT_CHUNKER

    @property
    def _chunker_dir(self) -> Path:
        """Where this chunker's index artifacts live.

        The default chunker keeps today's flat layout, byte-identical to before
        this field existed. Any other chunker gets its own subfolder, so two
        strategies can be ingested side by side without one overwriting the
        other's chunks, BM25 index or dense matrix.
        """
        if self.chunker == DEFAULT_CHUNKER:
            return self.artifacts_dir
        return self.artifacts_dir / self.chunker

    @property
    def chunks_path(self) -> Path:
        return self._chunker_dir / "chunks.jsonl"

    @property
    def embed_cache_path(self) -> Path:
        return self.artifacts_dir / "embed_cache.npz"

    @property
    def bm25_dir(self) -> Path:
        return self._chunker_dir / "bm25"

    @property
    def dense_path(self) -> Path:
        return self._chunker_dir / "dense.npz"

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
            # GCP_PROJECT is the fallback because some machines only set that one.
            vertex_project=(
                os.environ.get("VERTEX_PROJECT") or os.environ.get("GCP_PROJECT") or None
            ),
            vertex_location=os.environ.get("VERTEX_LOCATION") or DEFAULT_VERTEX_LOCATION,
            vertex_model=os.environ.get("VERTEX_MODEL", DEFAULT_VERTEX_MODEL),
            # Read from our own variable, not VERTEX_EMBEDDING_MODEL, so the .env
            # copied from the old repo cannot silently swap the model the gold
            # numbers were measured with.
            vertex_embed_model=os.environ.get(
                "FINRAG_VERTEX_EMBED_MODEL", DEFAULT_VERTEX_EMBED_MODEL
            ),
            vertex_rerank_model=os.environ.get(
                "FINRAG_VERTEX_RERANK_MODEL", DEFAULT_VERTEX_RERANK_MODEL
            ),
            embedder=os.environ.get("FINRAG_EMBEDDER", DEFAULT_EMBEDDER),
            chunker=os.environ.get("FINRAG_CHUNKER", DEFAULT_CHUNKER),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, resolved once on first use."""
    return Settings.from_env()
