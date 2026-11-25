# Breadwitness architecture

Breadwitness is a local image retrieval engine built from small modules with
one job each. 0.2 adds an embedding layer next to the 0.1 hash layer; the 0.1
modules were extended, not rewritten.

## Data flow

    image file -> ingestion -> preprocessing -> hashing -> ImageRecord
                                                              |
                   indexing (incremental, hashes) <-----------+
                         |                                    |
                         v                                    |
                   database (SQLite) <-- embedding_index <-- embedding (backend)
                         |                  (incremental)      |
                         v                                     |
                 embedding_store (float16 BLOBs)               |
                         |                                     |
                         +--> search --> similarity (cosine) <-+
                                |
                                v
                             ranking --> CLI

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Constants, supported extensions, path and model defaults |
| `models.py` | Dataclasses and enums (`MatchMethod`, `SearchMode`, reports) |
| `preprocessing.py` | Validate images, apply EXIF orientation, convert to RGB |
| `hashing.py` | SHA-256, pHash/dHash/aHash, Hamming distance |
| `ingestion.py` | Recursive file discovery; file -> `ImageRecord` |
| `database.py` | SQLite schema, migration, queries, statistics |
| `indexing.py` | Incremental hash indexing, change detection, pruning |
| `embedding.py` | Embedding backends and the backend registry |
| `embedding_store.py` | Persistent embedding storage and loading |
| `embedding_index.py` | Incremental embedding generation |
| `similarity.py` | Cosine similarity (chunked NumPy) |
| `search.py` | Exact, perceptual, embedding and hybrid retrieval |
| `ranking.py` | Scoring and evidence-tier ordering |
| `cli.py` | Typer commands |

## Search modes and ranking

Each result is classified by its strongest evidence:

1. **exact**: identical bytes (same SHA-256). Applies in every mode.
2. **perceptual**: pHash Hamming distance <= `--max-distance` (modes `hash`, `hybrid`).
3. **semantic**: embedding cosine >= `--min-similarity` (modes `embedding`, `hybrid`).

Results are ordered by tier first. Inside the perceptual tier: smaller pHash
distance, then higher cosine. Inside the semantic tier: higher cosine.

There is deliberately **no blended score**. Hash similarity and embedding
cosine are different quantities with different scales, and any weighting of
them would be arbitrary and would look more precise than it is. Each result
shows its raw signals (hash distances, cosine) and a `score` that is labeled
with its basis (`sha256`, `hash` or `embedding`). Scores with different bases
are not comparable; the tier decides the order. No score is a probability.

`auto` mode uses `hybrid` when embeddings exist for the selected model, and
`hash` otherwise, so a 0.1 index keeps working unchanged.

## Embedding storage

Embeddings live in the `embeddings` table of the same SQLite file, keyed by
`(image_id, model_id)`. Each row stores the dimension, the SHA-256 of the file
it was computed from, and the vector as float16 bytes.

* A vector is **current** only while its stored SHA-256 equals the image's, so
  a changed file never matches with an old vector, even before re-embedding.
* Several models can coexist. Switching model only embeds what that model is
  missing; `--purge-other-models` removes the rest.
* `ON DELETE CASCADE` removes embeddings when `index` prunes a file.
* At search time the current vectors are streamed into one preallocated
  float16 matrix and compared in chunks (float32 per chunk), so peak memory is
  about `images x dims x 2` bytes plus one chunk.

## Schema migration

Schema version 2 adds the `embeddings` table. A 0.1 database is migrated when
opened: the table is created and the version number updated. Existing `images`
rows are never modified or dropped. Opening a database from a newer
Breadwitness is refused with a clear error.

## Adding a backend

Implement `model_id`, `dim`, `default_min_similarity` and
`embed_images(images) -> (n, dim) float32 unit vectors`, then call
`register_backend("prefix", factory)`. Models are addressed as `prefix:arg`.

## Planned (not implemented)

FAISS or another approximate-nearest-neighbor index (search is currently an
exact linear scan), OCR, object detection, region matching, a learned ranking
model, external search providers and web discovery.
