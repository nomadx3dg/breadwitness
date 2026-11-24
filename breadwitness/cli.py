"""Command-line interface for Breadwitness."""

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from . import __version__
from .config import DEFAULT_LIMIT, DEFAULT_MAX_DISTANCE, HASH_BITS, resolve_db_path
from .database import Database, DatabaseError
from .indexing import index_directory
from .models import MatchMethod
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
) -> None:
    """Index a folder of images (incremental)."""
    db_path = resolve_db_path(db)

    def progress(file: Path, status: str) -> None:
        if verbose:
            typer.echo(f"  {status:<9} {file}")

    try:
        with Database(db_path) as database:
            report = index_directory(
                database, path, force=force, prune=prune, on_progress=progress
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
) -> None:
    """Find indexed images that match a query image."""
    with _open_existing_db(db) as database:
        try:
            response = search_image(
                database, query, limit=limit, max_distance=max_distance,
                exclude_self=exclude_self,
            )
        except InvalidImageError as exc:
            raise _fail(f"query is not a valid image: {exc}")

    q = response.query
    typer.echo(
        f"Query: {q.filename} ({q.width}x{q.height}, {q.format}) - "
        f"{response.candidates_scanned} indexed images scanned"
    )
    if not response.results:
        typer.echo(f"No matches within pHash distance {max_distance}.")
        return

    typer.echo(f"{len(response.results)} result(s):\n")
    for rank, result in enumerate(response.results, start=1):
        rec = result.record
        method = (
            "exact (identical file, SHA-256)"
            if result.method is MatchMethod.EXACT
            else "perceptual (hash similarity)"
        )
        typer.echo(f"{rank:>3}. {rec.filename}")
        typer.echo(f"     path:       {rec.path}")
        typer.echo(f"     dimensions: {rec.width}x{rec.height}")
        typer.echo(f"     method:     {method}")
        typer.echo(
            f"     pHash dist: {result.phash_distance}/{HASH_BITS} "
            f"(dHash {result.dhash_distance}, aHash {result.ahash_distance})"
        )
        typer.echo(f"     similarity: {result.score:.3f}")
    typer.echo(
        "\nSimilarity is a heuristic score from hash distances in [0, 1]. "
        "It is not a probability."
    )


@app.command()
def stats(db: DbOption = None) -> None:
    """Show statistics about the index."""
    with _open_existing_db(db) as database:
        info = database.stats()
    typer.echo(f"Database:         {resolve_db_path(db)}")
    typer.echo(f"Images indexed:   {info['total_images']}")
    typer.echo(f"Total size:       {_human_size(info['total_bytes'])}")
    formats = ", ".join(f"{k}: {v}" for k, v in info["formats"].items()) or "-"
    typer.echo(f"Formats:          {formats}")
    typer.echo(f"Duplicate groups: {info['duplicate_groups']}")
    typer.echo(f"Redundant copies: {info['redundant_copies']}")
    typer.echo(f"Last indexed:     {info['last_indexed_at'] or '-'}")


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
