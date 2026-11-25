# Breadwitness

**Every image leaves a trail.**

Breadwitness (Bread + Eyewitness) is an open-source, local-first visual
investigation and image retrieval engine. It is not a chatbot: it is a search
tool that finds the most accurate matching images in a collection you index.

> **Status: 0.2 (alpha).** Breadwitness searches **local folders only**. It
> does not crawl the web, query reverse-image-search services, read text in
> images, or detect objects. See [Roadmap](#roadmap).

## What 0.2 adds

- **Image embeddings.** Each image can be turned into a vector, stored in the
  index, and compared with a query image to find visually or semantically
  similar pictures (not just near-copies).
- **Hybrid search.** Exact (SHA-256), perceptual-hash and embedding signals are
  combined. Every signal stays visible in the results.
- **Incremental embedding index.** Unchanged files are skipped, changed files
  are re-embedded, switching models only embeds what is missing.
- **Two models.** OpenCLIP (semantic, optional install) and `color-hist`
  (built in, no download, color layout only).
- New commands `embed` and `models`; new options `--mode`, `--model`,
  `--min-similarity`, `--embed`. All 0.1 commands and options still work.

## Features

### Implemented

- Recursive scanning of JPG, JPEG, PNG, WebP, BMP; corrupt files reported and skipped
- EXIF orientation normalization
- SHA-256 file hashes; pHash, dHash, aHash perceptual hashes
- Persistent SQLite index; incremental indexing; changed/deleted file detection
- Exact-duplicate detection
- Exact, perceptual, embedding and hybrid search with ranked results
- Embedding generation (OpenCLIP or `color-hist`), CPU only, batched, resumable
- Embeddings stored as float16 in the same SQLite file; per-model storage
- Automatic, non-destructive migration of 0.1 indexes
- CLI: `index`, `embed`, `search`, `stats`, `duplicates`, `models`

### Planned (not implemented)

FAISS / approximate nearest-neighbor search, OCR and partial-text matching,
object detection, landmark recognition, region-based matching, a learned
ranking model, external reverse image search providers, public web image
discovery.

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/nomadx3dg/breadwitness.git
cd breadwitness
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core: hashes + color-hist, small install
```

### Semantic embeddings (OpenCLIP), CPU only

The default model needs PyTorch and OpenCLIP. No GPU is needed or used.
To avoid a large CUDA build of PyTorch, install the CPU build first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[embeddings]"
```

Without that extra, everything except OpenCLIP works, and you can use
`--model color-hist`.

### Model download and storage

OpenCLIP weights are downloaded **once, on first use**, from the Hugging Face
Hub, and cached (by default under `~/.cache/huggingface`). Set
`BREADWITNESS_MODEL_CACHE` to choose another folder. After that, Breadwitness
works offline. `color-hist` needs no download. Images never leave your machine.

## Usage

```bash
# 1. index (hashes) - same as 0.1
breadwitness index ./images

# 2. embed (incremental; first run downloads the model)
breadwitness embed
#    or do both at once:
breadwitness index ./images --embed

# 3. search
breadwitness search ./query.jpg
breadwitness search ./query.jpg --limit 20
breadwitness search ./query.jpg --mode embedding --min-similarity 0.75

breadwitness stats
```

Also available as `python -m breadwitness ...`. Use `--model color-hist`
(or `export BREADWITNESS_MODEL=color-hist`) to avoid installing OpenCLIP.

Example hybrid result:

```
Query: query.jpg (120x120, JPEG) - 5 indexed images scanned
Mode: hybrid (model: openclip:ViT-B-32/laion2b_s34b_b79k)
2 result(s):

  1. img2.png
     path:       /home/me/images/img2.png
     dimensions: 256x256
     method:     perceptual (hash similarity)
     pHash dist: 2/64 (dHash 1, aHash 3)
     embedding:  cosine 0.973
     similarity: 0.985 (basis: hash)
  2. img9.png
     ...
     method:     semantic (embedding similarity)
     pHash dist: 27/64 (dHash 25, aHash 24)
     embedding:  cosine 0.842
     similarity: 0.842 (basis: embedding)
```

(Illustrative output; actual numbers depend on your images and model.)

## CLI reference

| Command | Description |
| --- | --- |
| `index PATH` | Index a directory (recursive) or file; `--embed` also embeds |
| `embed` | Generate embeddings for indexed images (incremental) |
| `search QUERY` | Find matching indexed images |
| `stats` | Index statistics, including embedding coverage per model |
| `duplicates` | List groups of byte-identical files |
| `models` | List documented embedding models |

| Option | Applies to | Description |
| --- | --- | --- |
| `--db PATH` | all | Index database (`$BREADWITNESS_DB`, else `./.breadwitness/index.db`) |
| `--model SPEC` | `index --embed`, `embed`, `search` | Embedding model (`$BREADWITNESS_MODEL`, else the OpenCLIP default) |
| `--force`, `--prune/--no-prune`, `-v` | `index` | As in 0.1 |
| `--embed` | `index` | Embed after indexing |
| `--rebuild` | `embed` | Re-embed every image for this model |
| `--batch-size N` | `embed`, `index --embed` | Images per batch (default 16) |
| `--purge-other-models` | `embed` | Delete embeddings made by other models |
| `-n, --limit N` | `search` | Maximum results (default 10) |
| `--max-distance N` | `search` | Maximum pHash distance, 0-64 (default 10) |
| `--mode` | `search` | `auto` (default), `hash`, `embedding`, `hybrid` |
| `--min-similarity X` | `search` | Minimum cosine for semantic matches (-1 to 1) |
| `--exclude-self` | `search` | Ignore the query file if indexed |

`--mode auto` uses `hybrid` when embeddings exist for the model, otherwise
`hash`, which is exactly the 0.1 behavior.

### How results are ranked and scored

Results are classified as **exact** (identical bytes), **perceptual** (close
pHash) or **semantic** (close embedding), and ordered in that order. There is
no blended score: hash similarity and embedding cosine are different
quantities, so Breadwitness shows each signal and orders by evidence type
instead. The `similarity` value is labeled with its basis, values with
different bases are not comparable, and **no value is a probability** of two
images being the same. The default thresholds (`--max-distance 10`,
`--min-similarity` 0.80 for OpenCLIP, 0.85 for `color-hist`) are heuristics;
tune them on your own images.

## Maintaining the index

- **Skipping unchanged files:** `index` compares size and modification time;
  `embed` only processes images with no current vector for the model.
- **Changed files:** a vector is tied to the file's SHA-256. A changed file's
  old vector is ignored immediately and replaced on the next `embed`.
- **Missing files:** `index` prunes them (and their embeddings).
- **Corrupt or unsupported files:** reported and skipped, never fatal.
- **Changing model:** `embed --model NEW` embeds only what NEW lacks. Old
  vectors are kept until you pass `--purge-other-models`.
- **Upgrading from 0.1:** open your existing index with 0.2. The schema is
  upgraded in place; no image data is removed. Back up `index.db` first if the
  index is valuable (always sensible before any upgrade).
- Interrupted `embed` runs resume: work is committed per batch.

## Expected resource usage

| | OpenCLIP ViT-B-32 (laion2b) | color-hist |
| --- | --- | --- |
| Install | PyTorch + OpenCLIP (hundreds of MB; more with a CUDA build) | none extra |
| Model download | roughly 600 MB, once | none |
| RAM while embedding | about 1-2 GB (estimate) | tens of MB |
| Speed on CPU | a few to a few tens of images/s, hardware dependent (estimate) | hundreds/s |
| Vector size | 512 dims = 1 KB per image | 256 dims = 0.5 KB per image |
| Search RAM | about 1 KB x images (100k images = ~100 MB) | about 0.5 KB x images |

Storage and search-memory figures follow from the float16 storage format. The
speed, download and embedding-RAM figures are estimates; they have not been
benchmarked on your hardware. Search is an exact linear scan: fine for
hundreds of thousands of images, slower beyond that (FAISS is planned).

## Known limitations

- Local folders only. No web search, crawling, or reverse-image-search providers.
- Semantic matching quality depends on the model. OpenCLIP finds similar
  scenes and objects; it does not identify a specific place or person, and
  similar images are not evidence that two images share an origin.
- `color-hist` compares color layout only. It is a lightweight fallback, not a
  semantic model.
- Embeddings from different models are not comparable and are kept separate.
- The OpenCLIP backend was written against the OpenCLIP API but could not be
  exercised with real model weights in the development sandbox (no access to
  the model host); its logic is unit-tested only through the shared backend
  interface. Please report any problems on first real use.
- Perceptual hashes do not survive crops, rotations or mirroring.

## Architecture

See [`docs/architecture.md`](docs/architecture.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests need no network and no model download: they use the built-in
`color-hist` model and a tiny deterministic test model.

## Roadmap

1. **0.1:** hash-based local index and search (done)
2. **0.2:** embeddings and hybrid search (this release)
3. **0.3:** approximate nearest-neighbor index (FAISS) for large collections
4. **Later:** OCR and partial-text matching, object detection, region-based
   matching, landmark recognition, a learned ranking model, external reverse
   image search providers, public web image discovery

Roadmap items are plans, not commitments.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
