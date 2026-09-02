"""Chunker name to chunking function: the one place that knows what "heading" means.

`run_ingest` picks a chunker by a short string, the same way the retrieval side
picks a mode. It should not import chunking functions directly, or every new
strategy would mean edits in ingest, config and the CLI at once. Instead it
looks the name up in `CHUNKING_STRATEGIES` and calls what it finds.

Adding a strategy is two lines here: write the function, put it in the dict.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from finrag.ingest.chunker import chunk_markdown
from finrag.ingest.chunker_table import chunk_markdown_table_aware
from finrag.models import Chunk

# (path, doc_id, owner_dept, sensitivity) -> chunks, in document order. Exactly
# the signature of `chunk_markdown`, promoted to a type so callers can name it
# without importing a particular strategy.
ChunkerFn = Callable[[Path, str, str, str], list[Chunk]]

# The strategy every existing artifact was built with. Changing this default
# would change ingest output for everyone who has not set FINRAG_CHUNKER.
DEFAULT_CHUNKER = "heading"

# Adding a strategy is two lines: write the function, add it here.
CHUNKING_STRATEGIES: dict[str, ChunkerFn] = {
    "heading": chunk_markdown,
    "table_aware": chunk_markdown_table_aware,
}


def available_chunkers() -> list[str]:
    """Chunker names a caller may ask for, in a stable order for menus and help."""
    return sorted(CHUNKING_STRATEGIES)


def build_chunker(name: str) -> ChunkerFn:
    """Look up the chunking function for `name`, or say plainly which exist."""
    if name not in CHUNKING_STRATEGIES:
        known = ", ".join(available_chunkers())
        raise KeyError(f"unknown chunking strategy '{name}'. Known strategies: {known}")
    return CHUNKING_STRATEGIES[name]
