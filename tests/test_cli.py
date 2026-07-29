"""The `finrag` command line, driven in process against a scratch artifacts dir.

These tests call `cli.main` the way the console script does, with the settings
environment variables pointed at `tmp_path`, so nothing here writes into the
real `artifacts/` or `evals/results/`. Everything is offline: the naive mode is
plain word overlap, and no test needs credentials or a network.

Two things are being pinned.

  * `finrag eval` refuses to publish numbers it cannot trust. Missing chunks, an
    unknown mode and a gold set that no longer matches the corpus all exit 1
    with a message that says what to do, instead of printing a table built on
    bad labels.
  * ingest and eval work end to end on the real corpus and the real gold set,
    which is the one check that the whole R0 pipeline is actually wired up.
  * `finrag sweep` runs the leakage gate over the real corpus and exits 0 only
    when nothing leaked, which is the one check that the gate is wired up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from finrag import cli
from finrag.retrieval.registry import available_modes


def point_settings_at(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    work_dir: Path,
    gold_path: Path | None = None,
) -> None:
    """Send every writable path into `work_dir`, keeping the real inputs.

    Corpus, policy and gold set stay the shipped ones unless a test hands over
    its own gold file. Chunks and results go to the scratch directory.
    """
    monkeypatch.setenv("FINRAG_CORPUS_DIR", str(repo_root / "data" / "corpus"))
    monkeypatch.setenv("FINRAG_POLICY_PATH", str(repo_root / "policy" / "policy.yaml"))
    monkeypatch.setenv("FINRAG_ARTIFACTS_DIR", str(work_dir / "artifacts"))
    monkeypatch.setenv("FINRAG_RESULTS_DIR", str(work_dir / "results"))

    if gold_path is None:
        gold_path = repo_root / "evals" / "gold" / "gold_set.json"
    monkeypatch.setenv("FINRAG_GOLD_PATH", str(gold_path))


def read_the_one_report(results_dir: Path) -> dict[str, Any]:
    """The single JSON an eval run left behind."""
    written = sorted(results_dir.glob("*.json"))
    assert len(written) == 1, f"expected one report, found {written}"
    return json.loads(written[0].read_text(encoding="utf-8"))


def test_help_names_every_retrieval_mode(capsys: pytest.CaptureFixture[str]) -> None:
    """A person who does not know the mode names can find them from --help."""
    with pytest.raises(SystemExit) as exited:
        cli.main(["eval", "--help"])

    assert exited.value.code == 0
    printed = capsys.readouterr().out
    for mode in available_modes():
        assert mode in printed


def test_eval_before_ingest_says_to_run_ingest(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    point_settings_at(monkeypatch, repo_root, tmp_path)

    assert cli.main(["eval", "--mode", "naive"]) == 1
    assert "finrag ingest" in capsys.readouterr().out


def test_eval_refuses_an_unknown_mode_and_lists_the_known_ones(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    point_settings_at(monkeypatch, repo_root, tmp_path)
    assert cli.main(["ingest"]) == 0
    capsys.readouterr()

    assert cli.main(["eval", "--mode", "quantum"]) == 1

    printed = capsys.readouterr().out
    assert "quantum" in printed
    for mode in available_modes():
        assert mode in printed


def test_eval_refuses_a_gold_set_that_no_longer_matches_the_corpus(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A drifted gold set must stop the run, not quietly score against nonsense.

    We break a copy of the real file three ways: a tier that does not exist, a
    gold document that is not in the corpus, and a missing required field. The
    validator is meant to report all of them in one go.
    """
    document = json.loads((repo_root / "evals" / "gold" / "gold_set.json").read_text())
    broken = document["questions"][0]
    broken["tier"] = "T9"
    broken["gold_docs"] = ["finance/a_document_that_was_deleted"]
    del broken["notes"]

    broken_path = tmp_path / "broken_gold.json"
    broken_path.write_text(json.dumps(document), encoding="utf-8")

    point_settings_at(monkeypatch, repo_root, tmp_path, gold_path=broken_path)
    assert cli.main(["ingest"]) == 0
    capsys.readouterr()

    assert cli.main(["eval", "--mode", "naive"]) == 1

    printed = capsys.readouterr().out
    assert "gold set is invalid" in printed
    assert "tier 'T9' is not one of" in printed
    assert "finance/a_document_that_was_deleted" in printed
    assert "missing field 'notes'" in printed
    # Nothing was published from a gold set we do not trust.
    assert not (tmp_path / "results").exists()


def test_eval_refuses_a_gold_file_that_is_not_a_question_paper(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Valid JSON of the wrong shape is caught by the loader, not by a KeyError."""
    bad_path = tmp_path / "not_a_gold_set.json"
    bad_path.write_text(json.dumps([{"id": "q1"}]), encoding="utf-8")

    point_settings_at(monkeypatch, repo_root, tmp_path, gold_path=bad_path)
    assert cli.main(["ingest"]) == 0
    capsys.readouterr()

    assert cli.main(["eval", "--mode", "naive"]) == 1
    assert "expected a JSON object" in capsys.readouterr().out


def test_ingest_then_eval_runs_end_to_end_on_the_real_corpus(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole R0 pipeline in two commands, offline, on naive mode.

    This is the test that would catch a break between the chunker, the policy
    stamp, the retriever, the gold set and the harness, since each of those has
    its own unit tests but nothing else runs them in one line.
    """
    point_settings_at(monkeypatch, repo_root, tmp_path)

    assert cli.main(["ingest"]) == 0
    ingest_output = capsys.readouterr().out
    assert "chunks.jsonl" in ingest_output
    assert (tmp_path / "artifacts" / "chunks.jsonl").exists()

    assert cli.main(["eval", "--mode", "naive", "--rung", "r0"]) == 0
    eval_output = capsys.readouterr().out

    # The printed table leads with the section columns, which is the headline.
    assert "sec hit" in eval_output
    assert "sec mrr" in eval_output
    assert "OVERALL" in eval_output
    assert "total leakage: 0" in eval_output

    report = read_the_one_report(tmp_path / "results")
    assert report["rung"] == "r0"
    assert report["mode"] == "naive"
    assert report["n_questions"] == 50
    assert report["overall"]["leakage_count"] == 0
    assert report["overall"]["section_mrr"] is not None
    assert sorted(report["per_tier"]) == ["T1", "T2", "T3", "T4", "T5", "T6"]
    # Every T6 policy probe has to pass at every rung, R0 included.
    assert report["policy_probes"]["pass_rate"] == 1.0


def test_sweep_before_ingest_says_to_run_ingest(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    point_settings_at(monkeypatch, repo_root, tmp_path)

    assert cli.main(["sweep"]) == 1
    assert "finrag ingest" in capsys.readouterr().out


def test_sweep_refuses_an_unknown_mode_and_lists_the_known_ones(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    point_settings_at(monkeypatch, repo_root, tmp_path)
    assert cli.main(["ingest"]) == 0
    capsys.readouterr()

    assert cli.main(["sweep", "--modes", "quantum"]) == 1
    assert "quantum" in capsys.readouterr().out


def test_sweep_on_the_real_corpus_reports_no_leakage(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The gate itself, on the shipped corpus and gold set, in naive mode.

    One mode rather than all five, because the suite has to stay fast and the
    other four are swept in `test_sweep.py` over the fixture corpus. What this
    pins is the wiring: real chunks, real policy, all 50 questions, all 6 roles.
    """
    point_settings_at(monkeypatch, repo_root, tmp_path)
    assert cli.main(["ingest"]) == 0
    capsys.readouterr()

    assert cli.main(["sweep", "--modes", "naive", "--k", "5"]) == 0
    printed = capsys.readouterr().out
    assert "VIOLATIONS:      0" in printed

    report = read_the_one_report(tmp_path / "results")
    assert report["rung"] == "r4"
    assert report["mode"] == "sweep"
    assert report["cells_checked"] == 50 * 6
    assert report["violations"] == 0
    assert report["hits_inspected"] > 0
