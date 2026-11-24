# Breadwitness architecture

Breadwitness 0.1 is a local image retrieval engine built from small modules
with one job each. Everything above `database.py` is stateless.

## Data flow

    image file ──► ingestion ──► preprocessing ──► hashing ──► ImageRecord
                                                                  │
                      indexing (incremental) ◄────────────────────┤
                            │                                     │
                            ▼                                     ▼
                      database (SQLite)  ──────►  search ──► ranking ──► CLI

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Constants, supported extensions, database path resolution |
| `models.py` | Dataclasses: `ImageRecord`, `SearchResult`, `IndexReport`, `MatchMethod` |
| `preprocessing.py` | Validate images, apply EXIF orientation, convert to RGB |
| `hashing.py` | SHA-256, pHash/dHash/aHash, Hamming distance (scalar and NumPy) |
| `ingestion.py` | Recursive file discovery; file -> `ImageRecord` |
| `database.py` | SQLite schema, upserts, queries, statistics |
| `indexing.py` | Incremental indexing, change detection, pruning |
| `search.py` | Exact and perceptual retrieval |
| `ranking.py` | Scoring and ordering |
| `cli.py` | Typer commands |

## Matching

* **Exact**: the query file's SHA-256 equals an indexed file's SHA-256.
  These are byte-identical files.
* **Perceptual**: the 64-bit pHash Hamming distance is at most
  `--max-distance` (default 10). dHash and aHash distances are reported and
  contribute to the score.
* **Score**: `1 - (0.5*pHash + 0.3*dHash + 0.2*aHash) / 64`, capped at 0.99 for
  perceptual matches. 1.0 is reserved for exact matches. The score is a
  heuristic, not a probability.
* **Order**: exact matches first, then smaller pHash distance, then higher
  score, then path.

## Incremental indexing

A file is unchanged when its size and `mtime_ns` equal the stored values.
Otherwise it is re-hashed. If the SHA-256 changed the file counts as updated.
Files that disappear are pruned (only under the indexed directory); files that
become corrupt lose their stale record.

## Storage

SQLite, one `images` table plus a `meta` table holding `schema_version`.
Hashes are stored as 16-character hex strings. Search loads all fingerprints
and compares them with vectorized NumPy operations. This is a linear scan,
fine for hundreds of thousands of images, and is the part FAISS will replace.

## Extension points (planned, not implemented)

New recognition modules are meant to be added without touching existing ones:

* **Per-module tables**: a module such as embeddings or OCR stores its data in
  its own table keyed by `images.id`, with its own schema version in `meta`.
* **Per-module retrieval**: a retriever takes a query image and returns
  `SearchResult`-like hits. `MatchMethod` gains new members (for example
  `EMBEDDING`, `TEXT`, `REGION`).
* **Ranking**: `ranking.py` is the single place where hits from different
  retrievers are merged, and where a learned ranking model would plug in.
* **Ingestion hooks**: heavy analysis (OCR, detection) runs after the cheap
  hashing step and only for new or changed files.
