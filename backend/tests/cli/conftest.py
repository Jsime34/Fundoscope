"""The command-line tests never reach the real database."""

import pytest

import fundoscope.cli.history as history_cli
from fundoscope.db import MemoryValleyStore
from fundoscope.pipeline import CountedValleys


@pytest.fixture(autouse=True)
def no_database(tmp_path, monkeypatch):
    """`fundoscope history` finds no counted valley unless a test gives it one."""
    valleys = CountedValleys(tmp_path / "valleys", MemoryValleyStore())
    monkeypatch.setattr(history_cli, "default_valleys", lambda: valleys)
