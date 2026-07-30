"""The HTTP layer: status codes, and nothing leaking into a response body.

Every test builds the app over the six-chunk `micro_corpus` fixture with
`FakeLLM`, so there is no artifact on disk and no network in play. The leakage
tests search the whole serialised response, not one field, because a leak that
hides in `debug` or in an error message is still a leak.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from finrag.api.app import create_app
from finrag.generate import Answerer
from finrag.llm.fake import FakeLLM, ScriptedLLM
from finrag.policy.engine import PolicyEngine
from finrag.retrieval.registry import IndexBundle

SPEND_QUESTION = "what was the campaign spend and impressions"
REVENUE_QUESTION = "what was Q4 revenue"

# The finance chunk only c_level may see, and the PII held in the HR row payload.
DENIED_FINANCE_CHUNK = "finance/quarterly_financial_report#q4-revenue"
ROW_PII_NAME = "Aadhya Patel"
ROW_PII_SALARY = "1654767"


def make_client(chunks, policy_path=None, llm=None) -> TestClient:
    """An app over an in-memory corpus. Pass `policy_path` to enable 403s."""
    policy = PolicyEngine.from_yaml(policy_path) if policy_path else None
    answerer = Answerer(
        bundle=IndexBundle(chunks=list(chunks)),
        policy=policy,
        llm=llm if llm is not None else FakeLLM(),
    )
    return TestClient(create_app(answerer))


def test_health_reports_the_corpus_size_and_the_backend(micro_corpus) -> None:
    client = make_client(micro_corpus)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] == 6
    assert body["llm"] == "fake"


def test_the_mode_list_comes_from_the_retrieval_registry(micro_corpus) -> None:
    client = make_client(micro_corpus)
    body = client.get("/modes").json()
    assert "bm25" in body["modes"]
    assert body["default"] in body["modes"]


def test_the_role_list_comes_from_the_policy_file(micro_corpus, policy_path) -> None:
    client = make_client(micro_corpus, policy_path=policy_path)
    assert client.get("/roles").json()["roles"] == [
        "finance",
        "marketing",
        "hr",
        "engineering",
        "c_level",
        "employee",
    ]


def test_a_permitted_role_gets_an_answer_with_citations(micro_corpus, policy_path) -> None:
    client = make_client(micro_corpus, policy_path=policy_path)
    response = client.post(
        "/ask",
        json={"question": SPEND_QUESTION, "role": "marketing", "mode": "bm25", "k": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reason"] == "ok"
    assert body["answer"]
    assert body["citations"]


def test_the_react_mode_answers_with_the_agents_own_words_and_citations(
    micro_corpus, policy_path
) -> None:
    # A scripted backend, so the loop really runs offline: one search, then a
    # finish citing observation 1. The answer in the body is the agent's, and no
    # second generation call was made to produce it.
    llm = ScriptedLLM(
        [
            json.dumps({"thought": "find the spend", "action": "search", "argument": "spend"}),
            json.dumps(
                {
                    "thought": "that is enough",
                    "action": "finish",
                    "argument": "Spend was 2.0 million dollars [1].",
                }
            ),
        ]
    )
    client = make_client(micro_corpus, policy_path=policy_path, llm=llm)

    response = client.post(
        "/ask",
        json={"question": SPEND_QUESTION, "role": "marketing", "mode": "react", "k": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["reason"] == "ok"
    assert body["answer"] == "Spend was 2.0 million dollars [1]."
    assert body["citations"] == [body["hits"][0]["chunk_id"]]
    assert len(llm.calls) == 2
    assert DENIED_FINANCE_CHUNK not in json.dumps(body)


def test_a_role_the_policy_gives_nothing_is_refused_with_the_rule(
    micro_corpus, policy_path
) -> None:
    client = make_client(micro_corpus, policy_path=policy_path)
    response = client.post(
        "/ask",
        json={"question": SPEND_QUESTION, "role": "employee", "mode": "bm25"},
    )
    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail["reason"] == "policy_denied"
    assert "employee_non_general_deny" in detail["rules"]


def test_an_unknown_role_is_refused(micro_corpus, policy_path) -> None:
    client = make_client(micro_corpus, policy_path=policy_path)
    response = client.post(
        "/ask",
        json={"question": SPEND_QUESTION, "role": "ceo_of_everything"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["reason"] == "unknown_role"


def test_an_unknown_mode_is_refused(micro_corpus, policy_path) -> None:
    client = make_client(micro_corpus, policy_path=policy_path)
    response = client.post(
        "/ask",
        json={"question": SPEND_QUESTION, "role": "marketing", "mode": "telepathy"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["reason"] == "unknown_mode"


def test_graphrag_with_the_server_down_is_a_503_not_a_bare_500(
    micro_corpus, policy_path, monkeypatch
) -> None:
    monkeypatch.setattr("finrag.graph.store.bolt_is_open", lambda *a, **kw: False)
    client = make_client(micro_corpus, policy_path=policy_path)
    response = client.post(
        "/ask",
        json={"question": SPEND_QUESTION, "role": "marketing", "mode": "graphrag"},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["reason"] == "graph_unavailable"
    assert "docker start finrag-neo4j" in response.json()["detail"]["message"]


def test_an_empty_question_is_refused(micro_corpus, policy_path) -> None:
    client = make_client(micro_corpus, policy_path=policy_path)
    response = client.post("/ask", json={"question": "", "role": "marketing"})
    assert response.status_code == 422


def test_a_question_with_no_permitted_hit_answers_nothing_rather_than_guessing(
    micro_corpus, policy_path
) -> None:
    client = make_client(micro_corpus, policy_path=policy_path)
    body = client.post(
        "/ask",
        json={"question": "kubernetes autoscaling rollout", "role": "marketing"},
    ).json()
    assert body["reason"] == "no_answer"
    assert body["answer"] is None
    assert body["hits"] == []


def test_a_denied_chunk_id_never_appears_in_a_response(micro_corpus, policy_path) -> None:
    client = make_client(micro_corpus, policy_path=policy_path)
    response = client.post(
        "/ask",
        json={"question": REVENUE_QUESTION, "role": "marketing", "k": 5},
    )
    assert DENIED_FINANCE_CHUNK not in json.dumps(response.json())


def test_the_same_question_as_c_level_does_return_the_finance_chunk(
    micro_corpus, policy_path
) -> None:
    # The other half of the leakage test: the chunk is findable, so its absence
    # for marketing is the mask working, not the query failing to match.
    client = make_client(micro_corpus, policy_path=policy_path)
    response = client.post(
        "/ask",
        json={"question": REVENUE_QUESTION, "role": "c_level", "k": 5},
    )
    assert DENIED_FINANCE_CHUNK in json.dumps(response.json())


def test_hr_row_pii_never_appears_in_a_response(micro_corpus, policy_path) -> None:
    # HR may retrieve its own rows, so the hit comes back. The name and the salary
    # live in the row payload, which no response body carries.
    client = make_client(micro_corpus, policy_path=policy_path)
    body = json.dumps(
        client.post(
            "/ask",
            json={"question": "which employee works in finance as an analyst", "role": "hr"},
        ).json()
    )
    assert ROW_PII_NAME not in body
    assert ROW_PII_SALARY not in body
