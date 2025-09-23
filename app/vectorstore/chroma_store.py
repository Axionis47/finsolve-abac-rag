from __future__ import annotations
from typing import List, Dict, Any, Optional
import hashlib

try:
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore
    Settings = None  # type: ignore


def _require_chroma():
    if chromadb is None:
        raise RuntimeError("chromadb package not installed. Please install chromadb.")


def get_client(persist_dir: Optional[str] = None):
    _require_chroma()
    # Use dedicated client types to avoid global SharedSystemClient settings conflicts across tests
    if persist_dir:
        # Persistent client isolates storage by path
        return chromadb.PersistentClient(path=persist_dir)
    # Ephemeral in-memory client
    return chromadb.EphemeralClient()


def get_or_create_collection(client, name: str = "kb_main"):
    # cosine similarity
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def make_chunk_id(source_path: str, section_path: str, text: str) -> str:
    h = hashlib.md5()
    h.update(source_path.encode("utf-8"))
    h.update(b"|")
    h.update(section_path.encode("utf-8"))
    h.update(b"|")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _roles_to_flags(roles: List[str]) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    for r in roles:
        flags[f"role_{r}"] = True
    flags["allowed_roles_str"] = ",".join(sorted(roles))
    return flags


def upsert_chunks(collection, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
    ids = [c.get("id") or make_chunk_id(c["source_path"], c.get("section_path", ""), c.get("text", "")) for c in chunks]
    documents = [c.get("text", "") for c in chunks]
    metadatas = []
    for c in chunks:
        roles = c.get("allowed_roles", []) or []
        md = {
            "source_path": c.get("source_path"),
            "section_path": c.get("section_path"),
            "owner_dept": c.get("owner_dept"),
            "doc_type": c.get("doc_type"),
            "sensitivity": c.get("sensitivity"),
            **_roles_to_flags(list(roles)),
        }
        metadatas.append(md)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(ids)


def query(collection, query_embedding: List[float], top_k: int = 5, role: Optional[str] = None) -> List[Dict[str, Any]]:
    # Server-side filter using boolean flags
    where = None
    if role:
        where = {f"role_{role}": True}
    res = collection.query(query_embeddings=[query_embedding], n_results=top_k * 5, where=where)
    out: List[Dict[str, Any]] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    for text, md in zip(docs, metas):
        out.append({"text": text, **(md or {})})
    return out[:top_k]

