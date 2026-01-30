"""
Shared test fixtures and configuration for the ABAC RAG Chatbot test suite.

This module provides reusable fixtures for:
- Test client setup
- Authentication helpers
- Mock embeddings and LLM responses
- Temporary test corpus creation
- Cache reset between tests
- PDP and policy fixtures
"""
import sys
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional

import pytest

# Ensure repository root is on sys.path for imports like `from app.policy.pdp import PDP`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# Authentication Fixtures
# ============================================================================

# Test users matching app/routes/deps.py
TEST_USERS: Dict[str, Dict[str, str]] = {
    "tony": {"password": "password123", "role": "engineering"},
    "bruce": {"password": "securepass", "role": "marketing"},
    "sam": {"password": "financepass", "role": "finance"},
    "peter": {"password": "pete123", "role": "engineering"},
    "sid": {"password": "sidpass123", "role": "marketing"},
    "natasha": {"password": "hrpass123", "role": "hr"},
    "clark": {"password": "chief", "role": "c_level"},
}


def basic_auth(username: str, password: str) -> Dict[str, str]:
    """Create Basic Auth header from username and password."""
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


@pytest.fixture
def test_users() -> Dict[str, Dict[str, str]]:
    """Return test users dictionary."""
    return TEST_USERS.copy()


@pytest.fixture
def auth_engineering() -> Dict[str, str]:
    """Auth header for engineering user (tony)."""
    return basic_auth("tony", "password123")


@pytest.fixture
def auth_marketing() -> Dict[str, str]:
    """Auth header for marketing user (bruce)."""
    return basic_auth("bruce", "securepass")


@pytest.fixture
def auth_finance() -> Dict[str, str]:
    """Auth header for finance user (sam)."""
    return basic_auth("sam", "financepass")


@pytest.fixture
def auth_hr() -> Dict[str, str]:
    """Auth header for HR user (natasha)."""
    return basic_auth("natasha", "hrpass123")


@pytest.fixture
def auth_c_level() -> Dict[str, str]:
    """Auth header for c_level user (clark)."""
    return basic_auth("clark", "chief")


# ============================================================================
# Test Client Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def app_instance():
    """Create FastAPI app instance for testing."""
    from app.main import app
    return app


@pytest.fixture
def test_client(app_instance):
    """Create test client for the application."""
    from fastapi.testclient import TestClient
    return TestClient(app_instance)


# ============================================================================
# Mock Embeddings and LLM
# ============================================================================

def stub_embed_texts(texts: List[str], use_cache: bool = True, **kwargs) -> List[List[float]]:
    """
    Deterministic embeddings without network calls.

    Returns consistent embeddings based on text hash for reproducible tests.
    """
    out = []
    for t in texts:
        # Create deterministic vector from text hash
        s = sum(ord(c) for c in t) % 97
        vec = [float((s + i * 3) % 13) / 13.0 for i in range(16)]
        out.append(vec)
    return out


def stub_embed_texts_768(texts: List[str], use_cache: bool = True, **kwargs) -> List[List[float]]:
    """
    Deterministic 768-dimension embeddings for Vertex AI compatibility.
    """
    out = []
    for t in texts:
        s = sum(ord(c) for c in t) % 97
        vec = [float((s + i * 3) % 13) / 13.0 for i in range(768)]
        out.append(vec)
    return out


def fake_generate_answer(
    query: str,
    snippets: List[Dict[str, Any]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Fake LLM response for testing.

    Returns a predictable answer with citations based on input snippets.
    """
    if not snippets:
        return "I don't know based on the available context."

    first = snippets[0]
    src = f"{first.get('source_path', '')}#{first.get('section_path', '')}".strip('#')
    return f"ANSWER for: {query}. See [1].\n\nCitations:\n[1] {src}"


def fake_llm_generate(messages: List[Dict[str, str]], **kwargs) -> str:
    """Fake LLM generate function that returns predictable responses."""
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    return f"Mock response for: {user_msg[:50]}..."


@pytest.fixture
def mock_embeddings(monkeypatch):
    """Monkeypatch all embedding functions to use stub."""
    import app.ingest.dense_indexer as dense_indexer
    import app.routes.chat as chat_mod
    import app.routes.search as search_mod

    monkeypatch.setattr(dense_indexer, "embed_texts", stub_embed_texts)
    monkeypatch.setattr(chat_mod, "embed_texts", stub_embed_texts)
    monkeypatch.setattr(search_mod, "embed_texts", stub_embed_texts)

    # Also try to patch reranker if it exists
    try:
        import app.services.reranker_openai as reranker_mod
        monkeypatch.setattr(reranker_mod, "embed_texts", stub_embed_texts)
    except (ImportError, AttributeError):
        pass

    return stub_embed_texts


@pytest.fixture
def mock_llm(monkeypatch):
    """Monkeypatch LLM generation to use fake responses."""
    import app.services.generation as gen
    monkeypatch.setattr(gen, "generate_answer", fake_generate_answer)
    return fake_generate_answer


@pytest.fixture
def mock_all_external(mock_embeddings, mock_llm, monkeypatch):
    """Mock all external API calls (embeddings + LLM)."""
    # Mock chromadb client creation
    try:
        import app.routes.chat as chat_mod
        monkeypatch.setattr(chat_mod, "get_client", lambda *a, **k: object())
    except (ImportError, AttributeError):
        pass

    return {"embeddings": mock_embeddings, "llm": mock_llm}


# ============================================================================
# Temporary Corpus and Data Fixtures
# ============================================================================

@pytest.fixture
def temp_corpus(tmp_path) -> Path:
    """
    Create a temporary test corpus with sample documents.

    Returns the base directory path containing:
    - general/employee_handbook.md
    - engineering/tech_doc.md
    - finance/report.md
    - marketing/campaign.md
    - hr/hr_data.csv
    """
    base = tmp_path / "resources" / "data"

    # General - accessible to all
    (base / "general").mkdir(parents=True)
    (base / "general" / "employee_handbook.md").write_text(
        "# Employee Handbook\n\n"
        "## Leave Policies\n"
        "Employees receive 15-21 days of annual leave.\n\n"
        "## Benefits\n"
        "Health insurance and retirement plans are provided.\n",
        encoding="utf-8",
    )

    # Engineering - restricted
    (base / "engineering").mkdir(parents=True)
    (base / "engineering" / "tech_doc.md").write_text(
        "# Engineering Documentation\n\n"
        "## Architecture\n"
        "System uses microservices architecture.\n\n"
        "## APIs\n"
        "RESTful APIs with JSON payloads.\n",
        encoding="utf-8",
    )

    # Finance - restricted
    (base / "finance").mkdir(parents=True)
    (base / "finance" / "report.md").write_text(
        "# Financial Report Q4 2024\n\n"
        "## Revenue\n"
        "Total revenue: $10M\n\n"
        "## Expenses\n"
        "Operating expenses: $6M\n",
        encoding="utf-8",
    )

    # Marketing - restricted
    (base / "marketing").mkdir(parents=True)
    (base / "marketing" / "campaign.md").write_text(
        "# Marketing Campaign 2024\n\n"
        "## Growth Strategy\n"
        "Focus on digital channels and partnerships.\n\n"
        "## ROI\n"
        "Expected ROI of 150%.\n",
        encoding="utf-8",
    )

    # HR - restricted with PII
    (base / "hr").mkdir(parents=True)
    (base / "hr" / "hr_data.csv").write_text(
        "name,email,department,salary,performance_rating\n"
        "John Doe,john@example.com,engineering,100000,4.5\n"
        "Jane Smith,jane@example.com,marketing,95000,4.2\n",
        encoding="utf-8",
    )

    return base


@pytest.fixture
def temp_chroma_dir(tmp_path) -> Path:
    """Create a temporary directory for ChromaDB."""
    chroma_dir = tmp_path / ".chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir


# ============================================================================
# Cache Fixtures
# ============================================================================

@pytest.fixture(autouse=False)
def reset_cache():
    """Reset cache state before and after test."""
    try:
        from app.cache import reset_rag_cache, reset_cache_adapter
        reset_rag_cache()
        reset_cache_adapter()
        yield
        reset_rag_cache()
        reset_cache_adapter()
    except ImportError:
        yield


@pytest.fixture
def rag_cache():
    """Get a fresh RAG cache instance."""
    try:
        from app.cache import get_rag_cache, reset_rag_cache
        reset_rag_cache()
        return get_rag_cache()
    except ImportError:
        pytest.skip("app.cache not available")


# ============================================================================
# Policy Fixtures
# ============================================================================

@pytest.fixture
def pdp():
    """Get a PDP instance with default policy."""
    from app.policy.pdp import PDP
    policy_path = ROOT / "docs" / "policy.yaml"
    return PDP(str(policy_path))


@pytest.fixture
def policy_fingerprint():
    """Get a policy fingerprint for testing."""
    try:
        from app.cache import compute_pf_for_subject
        return compute_pf_for_subject
    except ImportError:
        pytest.skip("app.cache not available")


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment variables for isolated testing."""
    # Remove potentially interfering env vars
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "VERTEX_PROJECT"]:
        monkeypatch.delenv(key, raising=False)

    # Set test defaults
    monkeypatch.setenv("CACHE_ENABLED", "true")
    monkeypatch.setenv("HR_AGGREGATE_MODE", "disabled")
    monkeypatch.setenv("HR_MASKED_ROWS_MODE", "disabled")

    return monkeypatch


# ============================================================================
# Utility Functions (exported for use in tests)
# ============================================================================

__all__ = [
    "TEST_USERS",
    "basic_auth",
    "stub_embed_texts",
    "stub_embed_texts_768",
    "fake_generate_answer",
    "fake_llm_generate",
]
