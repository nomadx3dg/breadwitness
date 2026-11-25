"""Command-line interface for Breadwitness."""

import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from . import __version__
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LIMIT,
    DEFAULT_MAX_DISTANCE,
    HASH_BITS,
    resolve_db_path,
)
from .database import Database, DatabaseError
from .embedding import (
    EmbeddingBackendError,
    describe_models,
    get_backend,
    resolve_model_spec,
)
from .embedding_index import build_embeddings
from .embedding_store import EmbeddingStore
from .indexing import index_directory
from .models import EmbedReport, MatchMethod, SearchMode
from .preprocessing import InvalidImageError
from .search import search_image

app = typer.Typer(
    name="breadwitness",
    help="Breadwitness: every image leaves a trail. Local image indexing and search.",
    no_args_is_help=True,
    add_completion=False,
)

DbOption = Annotated[
    Optional[Path],
    typer.Option(
        "--db",
        help="Path to the index database. Defaults to $BREADWITNESS_DB, "
        "then ./.breadwitness/index.db.",
    ),
]


ModelOption = Annotated[
    Optional[str],
    typer.Option(
        "--model",
        help="Embedding model (see 'breadwitness models'). Defaults to "
        "$BREADWITNESS_MODEL, then the OpenCLIP ViT-B-32 model.",
    ),
]

BatchOption = Annotated[
    int,
    typer.Option("--batch-size", min=1, help="Images embedded per batch."),
]


class ModeOption(str, Enum):
    auto = "auto"
    hash = "hash"
    embedding = "embedding"
    hybrid = "hybrid"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"breadwitness {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Breadwitness: every image leaves a trail."""


def _fail(message: str, code: int = 1) -> "typer.Exit":
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(code)


def _open_existing_db(db: Optional[Path]) -> Database:
    path = resolve_db_path(db)
    if not path.is_file():
        raise _fail(f"no index found at {path}. Run 'breadwitness index <folder>' first.")
    try:
        return Database(path)
    except DatabaseError as exc:
        raise _fail(str(exc))


def _get_backend(model: Optional[str]):
    try:
        return get_backend(model)
    except EmbeddingBackendError as exc:
        raise _fail(str(exc))


def _run_embedding(
    database: Database, backend, batch_size: int, force: bool, purge_other: bool
) -> EmbedReport:
    def progress(done: int, total: int) -> None:
        typer.echo(f"\r  embedding {done}/{total}", nl=False, err=True)
        if done == total:
            typer.echo("", err=True)

    try:
        return build_embeddings(
            database, backend, batch_size=batch_size, force=force,
            purge_other_models=purge_other, on_progress=progress,
        )
    except EmbeddingBackendError as exc:
        raise _fail(str(exc))


def _print_embed_report(report: EmbedReport) -> None:
    typer.echo(f"Embeddings for {report.model_id}")
    typer.echo(f"  images:      {report.total}")
    typer.echo(f"  embedded:    {report.embedded}")
    typer.echo(f"  up to date:  {report.up_to_date}")
    if report.purged:
        typer.echo(f"  purged:      {report.purged} vectors from other models")
    typer.echo(f"  skipped:     {len(report.skipped)}")
    for skipped_path, reason in report.skipped:
        typer.echo(f"    skipped {skipped_path}: {reason}", err=True)


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


@app.command()
def index(
    path: Annotated[
        Path,
        typer.Argument(exists=True, help="Image directory (scanned recursively) or file."),
    ],
    db: DbOption = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-process files even if unchanged.")
    ] = False,
    prune: Annotated[
        bool,
        typer.Option(
            "--prune/--no-prune",
            help="Remove index entries for files that no longer exist.",
        ),
    ] = True,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Print the status of every file.")
    ] = False,
    embed: Annotated[
        bool,
        typer.Option("--embed", help="Also generate embeddings after indexing."),
    ] = False,
    model: ModelOption = None,
    batch_size: BatchOption = DEFAULT_BATCH_SIZE,
) -> None:
    """Index a folder of images (incremental)."""
    db_path = resolve_db_path(db)
    backend = _get_backend(model) if embed else None  # fail fast on bad model
    embed_report = None

    def progress(file: Path, status: str) -> None:
        if verbose:
            typer.echo(f"  {status:<9} {file}")

    try:
        with Database(db_path) as database:
            report = index_directory(
                database, path, force=force, prune=prune, on_progress=progress
            )
            if backend is not None:
                embed_report = _run_embedding(
                    database, backend, batch_size, False, False
                )
    except DatabaseError as exc:
        raise _fail(str(exc))

    typer.echo(f"Indexed {path}")
    typer.echo(f"  database:  {db_path}")
    typer.echo(f"  scanned:   {report.scanned}")
    typer.echo(f"  added:     {report.added}")
    typer.echo(f"  updated:   {report.updated}")
    typer.echo(f"  unchanged: {report.unchanged}")
    typer.echo(f"  removed:   {report.removed}")
    typer.echo(f"  skipped:   {len(report.skipped)}")
    for skipped_path, reason in report.skipped:
        typer.echo(f"    skipped {skipped_path}: {reason}", err=True)
    if embed_report is not None:
        _print_embed_report(embed_report)


@app.command()
def embed(
    db: DbOption = None,
    model: ModelOption = None,
    rebuild: Annotated[
        bool,
        typer.Option("--rebuild", help="Re-embed every image for this model."),
    ] = False,
    batch_size: BatchOption = DEFAULT_BATCH_SIZE,
    purge_other_models: Annotated[
        bool,
        typer.Option(
            "--purge-other-models",
            help="Delete stored embeddings made by other models.",
        ),
    ] = False,
) -> None:
    """Generate embeddings for indexed images (incremental)."""
    with _open_existing_db(db) as database:
        backend = _get_backend(model)
        report = _run_embedding(database, backend, batch_size, rebuild, purge_other_models)
    _print_embed_report(report)


@app.command()
def models() -> None:
    """List documented embedding models."""
    typer.echo(f"Active model: {resolve_model_spec(None)}\n")
    for spec, description, available in describe_models():
        status = "available" if available else "not installed"
        typer.echo(f"  {spec}\n      {description} [{status}]")
    typer.echo(
        "\nOther OpenCLIP models work as openclip:<model>/<pretrained>. "
        "Embeddings from different models are kept separately."
    )


@app.command()
def search(
    query: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, help="Query image."),
    ],
    db: DbOption = None,
    limit: Annotated[
        int, typer.Option("--limit", "-n", min=1, help="Maximum number of results.")
    ] = DEFAULT_LIMIT,
    max_distance: Annotated[
        int,
        typer.Option(
            "--max-distance", min=0, max=HASH_BITS,
            help=f"Maximum pHash Hamming distance (0-{HASH_BITS}) for approximate matches.",
        ),
    ] = DEFAULT_MAX_DISTANCE,
    exclude_self: Annotated[
        bool,
        typer.Option("--exclude-self", help="Ignore the query file itself if it is indexed."),
    ] = False,
    mode: Annotated[
        ModeOption,
        typer.Option(
            "--mode",
            help="hash = exact + perceptual (0.1 behavior); embedding = exact + "
            "embedding similarity; hybrid = all; auto = hybrid if embeddings "
            "exist for the model, else hash.",
        ),
    ] = ModeOption.auto,
    model: ModelOption = None,
    min_similarity: Annotated[
        Optional[float],
        typer.Option(
            "--min-similarity", min=-1.0, max=1.0,
            help="Minimum embedding cosine similarity for semantic matches "
            "(default depends on the model).",
        ),
    ] = None,
) -> None:
    """Find indexed images that match a query image."""
    with _open_existing_db(db) as database:
        spec = resolve_model_spec(model)
        fresh = EmbeddingStore(database).count_fresh(spec)
        if mode is ModeOption.auto:
            effective = SearchMode.HYBRID if fresh else SearchMode.HASH
            if not fresh:
                typer.echo(
                    f"Note: no embeddings for {spec}; using hash matching. "
                    "Run 'breadwitness embed' to enable embedding search.",
                    err=True,
                )
        else:
            effective = SearchMode(mode.value)
            if effective is not SearchMode.HASH and not fresh:
                raise _fail(
                    f"no embeddings for {spec}. Run 'breadwitness embed"
                    f"{' --model ' + model if model else ''}' first."
                )
        backend = _get_backend(model) if effective is not SearchMode.HASH else None
        try:
            response = search_image(
                database, query, limit=limit, max_distance=max_distance,
                exclude_self=exclude_self, mode=effective, backend=backend,
                min_similarity=min_similarity,
            )
        except InvalidImageError as exc:
            raise _fail(f"query is not a valid image: {exc}")
        except ValueError as exc:
            raise _fail(str(exc))

    q = response.query
    typer.echo(
        f"Query: {q.filename} ({q.width}x{q.height}, {q.format}) - "
        f"{response.candidates_scanned} indexed images scanned"
    )
    if response.model_id and effective is not SearchMode.HASH:
        typer.echo(f"Mode: {effective.value} (model: {response.model_id})")
        if response.unembedded:
            typer.echo(
                f"Note: {response.unembedded} indexed images have no current embedding "
                "and could only match by hash. Run 'breadwitness embed'.",
                err=True,
            )
    if not response.results:
        typer.echo("No matches found.")
        return

    typer.echo(f"{len(response.results)} result(s):\n")
    for rank, result in enumerate(response.results, start=1):
        rec = result.record
        method = {
            MatchMethod.EXACT: "exact (identical file, SHA-256)",
            MatchMethod.PERCEPTUAL: "perceptual (hash similarity)",
            MatchMethod.SEMANTIC: "semantic (embedding similarity)",
        }[result.method]
        typer.echo(f"{rank:>3}. {rec.filename}")
        typer.echo(f"     path:       {rec.path}")
        typer.echo(f"     dimensions: {rec.width}x{rec.height}")
        typer.echo(f"     method:     {method}")
        typer.echo(
            f"     pHash dist: {result.phash_distance}/{HASH_BITS} "
            f"(dHash {result.dhash_distance}, aHash {result.ahash_distance})"
        )
        if result.cosine is not None:
            typer.echo(f"     embedding:  cosine {result.cosine:.3f}")
        typer.echo(
            f"     similarity: {result.score:.3f} (basis: {result.score_basis})"
        )
    typer.echo(
        "\nSimilarity is a heuristic value in [0, 1]. It is not a probability. "
        "Scores based on different signals are not comparable: results are "
        "ordered by evidence type first (exact, perceptual, semantic)."
    )


@app.command()
def stats(db: DbOption = None) -> None:
    """Show statistics about the index."""
    with _open_existing_db(db) as database:
        info = database.stats()
        embeddings = EmbeddingStore(database).model_counts()
    typer.echo(f"Database:         {resolve_db_path(db)}")
    typer.echo(f"Images indexed:   {info['total_images']}")
    typer.echo(f"Total size:       {_human_size(info['total_bytes'])}")
    formats = ", ".join(f"{k}: {v}" for k, v in info["formats"].items()) or "-"
    typer.echo(f"Formats:          {formats}")
    typer.echo(f"Duplicate groups: {info['duplicate_groups']}")
    typer.echo(f"Redundant copies: {info['redundant_copies']}")
    typer.echo(f"Last indexed:     {info['last_indexed_at'] or '-'}")
    if not embeddings:
        typer.echo("Embeddings:       none (run 'breadwitness embed')")
    for model_id, counts in embeddings.items():
        typer.echo(
            f"Embeddings:       {model_id}: {counts['fresh']}/{info['total_images']} "
            f"current, {counts['stale']} stale, {counts['dim']} dims"
        )


@app.command()
def duplicates(db: DbOption = None) -> None:
    """List groups of byte-identical files in the index."""
    with _open_existing_db(db) as database:
        groups = database.duplicate_groups()
    if not groups:
        typer.echo("No exact duplicates found.")
        return
    for number, group in enumerate(groups, start=1):
        typer.echo(f"Group {number} (sha256 {group[0].sha256[:12]}..., {len(group)} files):")
        for rec in group:
            typer.echo(f"  {rec.path}")


def main() -> None:
    """Console-script entry point."""
    app(prog_name="breadwitness")


if __name__ == "__main__":
    sys.exit(main())
