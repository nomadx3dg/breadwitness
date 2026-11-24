"""Central configuration and constants for Breadwitness."""

from __future__ import annotations

import os
from pathlib import Path

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

# Perceptual hashes are 8x8 = 64 bits, stored as 16 hex characters.
HASH_SIZE = 8
HASH_BITS = HASH_SIZE * HASH_SIZE

DEFAULT_LIMIT = 10
DEFAULT_MAX_DISTANCE = 10

DB_ENV_VAR = "BREADWITNESS_DB"
DEFAULT_DB_PATH = Path(".breadwitness") / "index.db"
SCHEMA_VERSION = 1

FILE_HASH_CHUNK_SIZE = 1024 * 1024


def resolve_db_path(db: Path | None = None) -> Path:
    """Pick the database path: explicit argument, then env var, then default."""
    if db is not None:
        return Path(db).expanduser()
    env_value = os.environ.get(DB_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_DB_PATH
