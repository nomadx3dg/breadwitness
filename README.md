# Breadwitness

**Every image leaves a trail.**

Breadwitness is an open-source, machine-learning-powered visual investigation
and image retrieval engine. It is not a chatbot: it is a search and
recognition tool that finds the most accurate matching images using computer
vision.

> **Status: 0.1 prototype (alpha).** Version 0.1 is a local image search
> engine built on hashing. It does **not** yet use machine learning, OCR,
> object detection or any online service. See [Roadmap](#roadmap).

## Features

### Implemented in 0.1

- Recursive scanning of JPG, JPEG, PNG, WebP and BMP files
- Image validation; corrupted files are reported and skipped
- EXIF orientation normalization
- SHA-256 file hashes and 64-bit perceptual hashes (pHash, dHash, aHash)
- Persistent SQLite index (survives restarts)
- Incremental indexing and detection of changed, new and deleted files
- Exact-duplicate detection
- Exact matching (identical bytes) and perceptual matching (Hamming distance)
- Ranked results with a configurable limit and distance threshold
- CLI: `index`, `search`, `stats`, `duplicates`
- CPU only, no API keys, no network access, no GPU

### Planned (not implemented)

OpenCLIP embeddings, FAISS vector search, OCR and partial-text matching,
object detection, building and landmark recognition, region-based matching,
bounding-box pointers, a custom ranking model, external reverse image search
providers, public web image discovery and a continuously growing index.

## Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/nomadx3dg/breadwitness.git
cd breadwitness
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```bash
breadwitness index ./images
breadwitness search ./query.jpg
breadwitness search ./query.jpg --limit 20
breadwitness stats
```

The same commands work through Python:

```bash
python -m breadwitness index ./images
python -m breadwitness search ./query.jpg
```

Example output:

```
Query: query.jpg (120x120, JPEG) - 5 indexed images scanned
1 result(s):

  1. img2.png
     path:       /home/me/images/img2.png
     dimensions: 256x256
     method:     perceptual (hash similarity)
     pHash dist: 0/64 (dHash 0, aHash 1)
     similarity: 0.990

Similarity is a heuristic score from hash distances in [0, 1]. It is not a probability.
```

A Python API example is in `examples/quickstart.py`.

## CLI reference

Global option: `--version`.

| Command | Description |
| --- | --- |
| `index PATH` | Index a directory (recursive) or a single image |
| `search QUERY` | Find indexed images matching a query image |
| `stats` | Show index statistics |
| `duplicates` | List groups of byte-identical files |

Options:

| Option | Applies to | Description |
| --- | --- | --- |
| `--db PATH` | all | Index database. Default: `$BREADWITNESS_DB`, else `./.breadwitness/index.db` |
| `--force` | `index` | Re-process files even if unchanged |
| `--prune/--no-prune` | `index` | Remove entries for deleted files (default: prune) |
| `-v, --verbose` | `index` | Print the status of every file |
| `-n, --limit N` | `search` | Maximum results (default 10) |
| `--max-distance N` | `search` | Maximum pHash Hamming distance, 0-64 (default 10) |
| `--exclude-self` | `search` | Ignore the query file if it is itself indexed |

Each result shows the filename, original path, dimensions, matching method
(`exact` or `perceptual`), pHash distance and similarity score.

### How to read results

- **exact**: the file is byte-identical to the query (same SHA-256).
- **perceptual**: the image looks similar. A pHash distance of 0 to about 10
  (out of 64) usually means the same picture after resizing or recompression,
  but it is not proof. Crops, heavy edits and unrelated images with similar
  layouts can also land close.
- **similarity** is a heuristic derived from hash distances. It is **not a
  probability** and should not be read as a confidence.

## Architecture

```
breadwitness/
  config.py         constants and database path resolution
  models.py         dataclasses shared by all modules
  preprocessing.py  validation, EXIF orientation, RGB conversion
  hashing.py        SHA-256, perceptual hashes, Hamming distance
  ingestion.py      file discovery and ImageRecord creation
  database.py       SQLite persistence
  indexing.py       incremental indexing
  search.py         exact and perceptual retrieval
  ranking.py        scoring and ordering
  cli.py            Typer commands
```

Details and the plan for adding recognition modules independently are in
[`docs/architecture.md`](docs/architecture.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests generate their own synthetic images; no downloads are needed.

## Known limitations

- Perceptual hashes find resized, recompressed and lightly edited copies. They
  do not understand content, so crops, rotations, mirrored images and
  different photos of the same scene are mostly not found.
- Search is a linear scan over all indexed images.
- Only local files are searched.

## Roadmap

1. **0.1 (this release):** hash-based local index and search.
2. **0.2:** OpenCLIP embeddings and FAISS for semantic similarity.
3. **0.3:** OCR and partial-text matching.
4. **0.4:** object detection and bounding-box pointers.
5. **0.5:** region-based matching (local features).
6. **Later:** building and landmark recognition, a custom ranking model,
   external reverse image search providers, public web image discovery and a
   continuously growing index.

Roadmap items are plans, not commitments.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
