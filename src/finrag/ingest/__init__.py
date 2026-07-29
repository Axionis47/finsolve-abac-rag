"""Ingest: turn the raw corpus into chunks the retrievers can index.

The markdown chunker lives here; the HR CSV reader, the HR aggregates, and the
pipeline that stitches them together join it in the same package.
"""

from finrag.ingest.chunker import chunk_markdown, extract_quarter

__all__ = ["chunk_markdown", "extract_quarter"]
