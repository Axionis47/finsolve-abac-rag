"""
Tests for audit logging utilities.
"""
import json
import os
import pytest
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from app.utils.audit import gen_correlation_id, log_event


class TestGenCorrelationId:
    """Tests for correlation ID generation."""

    def test_returns_string(self):
        """Returns a string."""
        cid = gen_correlation_id()
        assert isinstance(cid, str)

    def test_returns_hex_string(self):
        """Returns a valid hex string."""
        cid = gen_correlation_id()
        # Should be able to parse as hex
        int(cid, 16)

    def test_returns_32_char_string(self):
        """Returns a 32-character string (UUID hex)."""
        cid = gen_correlation_id()
        assert len(cid) == 32

    def test_generates_unique_ids(self):
        """Generates unique IDs on each call."""
        ids = [gen_correlation_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_no_dashes_in_id(self):
        """Generated ID has no dashes (hex only)."""
        cid = gen_correlation_id()
        assert "-" not in cid


class TestLogEvent:
    """Tests for event logging."""

    def test_logs_to_stdout(self, capsys):
        """Logs JSON to stdout."""
        cid = gen_correlation_id()
        log_event(cid, "test.event", {"key": "value"})

        captured = capsys.readouterr()
        # Should have logged something
        assert captured.out.strip()

        # Should be valid JSON
        record = json.loads(captured.out.strip())
        assert record["cid"] == cid
        assert record["event"] == "test.event"
        assert record["key"] == "value"

    def test_includes_timestamp(self, capsys):
        """Log record includes timestamp."""
        log_event("cid123", "test", {})

        captured = capsys.readouterr()
        record = json.loads(captured.out.strip())
        assert "ts" in record
        assert isinstance(record["ts"], (int, float))

    def test_merges_payload(self, capsys):
        """Payload is merged into log record."""
        payload = {"user": "alice", "action": "login", "ip": "127.0.0.1"}
        log_event("cid", "auth.login", payload)

        captured = capsys.readouterr()
        record = json.loads(captured.out.strip())
        assert record["user"] == "alice"
        assert record["action"] == "login"
        assert record["ip"] == "127.0.0.1"

    def test_writes_to_file(self, tmp_path, monkeypatch):
        """Writes to daily log file when AUDIT_LOG_DIR is set."""
        log_dir = tmp_path / "audit_logs"

        # Mock get_settings to return our test log dir
        def mock_settings():
            return {"AUDIT_LOG_DIR": str(log_dir)}

        monkeypatch.setattr("app.utils.audit.get_settings", mock_settings)

        cid = gen_correlation_id()
        log_event(cid, "file.test", {"data": "test"})

        # Check that a log file was created
        log_files = list(log_dir.glob("*.jsonl"))
        assert len(log_files) == 1

        # Check content
        content = log_files[0].read_text()
        record = json.loads(content.strip())
        assert record["cid"] == cid
        assert record["event"] == "file.test"

    def test_creates_log_directory(self, tmp_path, monkeypatch):
        """Creates log directory if it doesn't exist."""
        log_dir = tmp_path / "new_audit_dir" / "nested"

        def mock_settings():
            return {"AUDIT_LOG_DIR": str(log_dir)}

        monkeypatch.setattr("app.utils.audit.get_settings", mock_settings)

        log_event("cid", "test", {})

        assert log_dir.exists()

    def test_appends_to_existing_file(self, tmp_path, monkeypatch):
        """Appends to existing log file."""
        log_dir = tmp_path / "audit"

        def mock_settings():
            return {"AUDIT_LOG_DIR": str(log_dir)}

        monkeypatch.setattr("app.utils.audit.get_settings", mock_settings)

        # Log multiple events
        log_event("cid1", "event1", {})
        log_event("cid2", "event2", {})
        log_event("cid3", "event3", {})

        # Check file has all entries
        log_files = list(log_dir.glob("*.jsonl"))
        content = log_files[0].read_text()
        lines = [l for l in content.strip().split("\n") if l]
        assert len(lines) == 3

    def test_does_not_fail_on_stdout_error(self, monkeypatch):
        """Does not raise if stdout write fails."""
        def failing_write(*args, **kwargs):
            raise IOError("Write failed")

        # Mock stdout.write to fail
        mock_stdout = StringIO()
        mock_stdout.write = failing_write
        monkeypatch.setattr(sys, "stdout", mock_stdout)

        # Should not raise
        log_event("cid", "test", {})

    def test_does_not_fail_on_file_error(self, monkeypatch, capsys):
        """Does not raise if file write fails."""
        def mock_settings():
            return {"AUDIT_LOG_DIR": "/nonexistent/readonly/path"}

        monkeypatch.setattr("app.utils.audit.get_settings", mock_settings)

        # Should not raise even though path is invalid
        log_event("cid", "test", {})

        # Should still log to stdout
        captured = capsys.readouterr()
        assert captured.out.strip()

    def test_uses_utc_for_filename(self, tmp_path, monkeypatch):
        """Uses UTC date for log filename."""
        from datetime import datetime, timezone

        log_dir = tmp_path / "audit"

        def mock_settings():
            return {"AUDIT_LOG_DIR": str(log_dir)}

        monkeypatch.setattr("app.utils.audit.get_settings", mock_settings)

        log_event("cid", "test", {})

        # Filename should be today's date in UTC
        expected_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_files = list(log_dir.glob("*.jsonl"))
        assert len(log_files) == 1
        assert expected_date in log_files[0].name
