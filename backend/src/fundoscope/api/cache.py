"""A small on-disk cache of parcel histories, until the `db` module (FR-02) replaces it.

Key: hash of the uploaded file's bytes + the day. The same file uploaded again on the same
day is answered instantly and costs nothing; a new day may bring new satellite images, so
it is computed again. Files live in data/private/cache/, which is git-ignored.
"""

import hashlib
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from fundoscope.api.schemas import HistoryOut


def cache_key(data: bytes, day: date) -> str:
    return f"{hashlib.sha256(data).hexdigest()[:16]}-{day.isoformat()}"


def load(folder: Path, key: str) -> HistoryOut | None:
    path = folder / f"history-{key}.json"
    try:
        return HistoryOut.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):  # missing, unreadable, or from an older format
        return None


def save(folder: Path, key: str, result: HistoryOut) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"history-{key}.json").write_text(result.model_dump_json(), encoding="utf-8")
