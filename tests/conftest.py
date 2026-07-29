"""Shared fixtures. Everything here is in memory and offline."""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# The editable install can go missing (a stale or skipped .pth file), which
# would fail every test for a reason that has nothing to do with the code.
if find_spec("finrag") is None and str(SRC) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SRC))

from finrag.config import Settings  # noqa: E402
from finrag.llm.fake import FakeEmbeddings, FakeLLM  # noqa: E402
from finrag.models import Chunk  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def corpus_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "corpus"


@pytest.fixture(scope="session")
def policy_path(repo_root: Path) -> Path:
    return repo_root / "policy" / "policy.yaml"


@pytest.fixture(scope="session")
def settings(repo_root: Path) -> Settings:
    return Settings.from_env(root=repo_root)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def micro_corpus() -> list[Chunk]:
    """Six tiny chunks over three owner departments.

    Deliberately mirrors the real corpus in miniature: a finance risk that only
    a marketing doc quantifies, two near-identical quarterly marketing sections
    with colliding numbers, one HR row with PII kept out of the text, and one
    HR aggregate.
    """
    return [
        Chunk(
            chunk_id="finance/quarterly_financial_report#q4-revenue",
            doc_id="finance/quarterly_financial_report",
            section_path="Quarterly Financial Report > Q4 > Revenue",
            section_slug="q4-revenue",
            text=(
                "Quarterly Financial Report > Q4 > Revenue. "
                "Q4 revenue reached 2.6 billion dollars, up 12 percent."
            ),
            owner_dept="finance",
            doc_type="md",
            sensitivity="confidential",
            # Confidential finance markdown: the policy gives it to c_level only,
            # not even to finance. See test_policy.py, which asserts that.
            allowed_roles=("c_level",),
            quarter="Q4",
        ),
        Chunk(
            chunk_id="finance/quarterly_financial_report#q4-risks",
            doc_id="finance/quarterly_financial_report",
            section_path="Quarterly Financial Report > Q4 > Risks",
            section_slug="q4-risks",
            text=(
                "Quarterly Financial Report > Q4 > Risks. "
                "North America saturation is the leading risk; mitigation is "
                "covered by the Q4 regional campaign."
            ),
            owner_dept="finance",
            doc_type="md",
            sensitivity="confidential",
            # Confidential finance markdown: the policy gives it to c_level only,
            # not even to finance. See test_policy.py, which asserts that.
            allowed_roles=("c_level",),
            quarter="Q4",
        ),
        Chunk(
            chunk_id="marketing/market_report_q1_2024#campaign-spend",
            doc_id="marketing/market_report_q1_2024",
            section_path="Market Report Q1 2024 > Campaign Spend",
            section_slug="campaign-spend",
            text=(
                "Market Report Q1 2024 > Campaign Spend. "
                "Spend was 2.0 million dollars with 500 thousand impressions "
                "from events."
            ),
            owner_dept="marketing",
            doc_type="md",
            sensitivity="internal",
            allowed_roles=("marketing", "c_level"),
            quarter="Q1",
        ),
        Chunk(
            chunk_id="marketing/market_report_q3_2024#campaign-spend",
            doc_id="marketing/market_report_q3_2024",
            section_path="Market Report Q3 2024 > Campaign Spend",
            section_slug="campaign-spend",
            text=(
                "Market Report Q3 2024 > Campaign Spend. "
                "Spend was 2.0 million dollars with 500 thousand impressions "
                "from influencers."
            ),
            owner_dept="marketing",
            doc_type="md",
            sensitivity="internal",
            allowed_roles=("marketing", "c_level"),
            quarter="Q3",
        ),
        Chunk(
            chunk_id="hr/hr_data#row-FINEMP1000",
            doc_id="hr/hr_data",
            section_path="HR Data > Employee Records",
            section_slug="row-FINEMP1000",
            text=(
                "HR Data > Employee Records. Employee FINEMP1000 works in the "
                "Finance department at the Bengaluru office as an analyst."
            ),
            owner_dept="hr",
            doc_type="csv_row",
            sensitivity="restricted",
            allowed_roles=("hr", "c_level"),
            payload={
                "employee_id": "FINEMP1000",
                "full_name": "Aadhya Patel",
                "email": "aadhya.patel@finsolve.example",
                "salary": 1654767,
                "department": "Finance",
            },
        ),
        Chunk(
            chunk_id="hr/aggregates#salary-by-department",
            doc_id="hr/aggregates",
            section_path="HR Aggregates > Salary by Department",
            section_slug="salary-by-department",
            text=(
                "HR Aggregates > Salary by Department. Average salary is "
                "highest in Risk at 1,654,767 rupees; the company mean is "
                "1,187,565 rupees."
            ),
            owner_dept="hr",
            doc_type="aggregate",
            sensitivity="internal",
            allowed_roles=("hr", "c_level"),
        ),
    ]
