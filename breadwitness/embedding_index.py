"""Incremental generation of embeddings for indexed images."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .config import DEFAULT_BATCH_SIZE
from .database import Database
from .embedding import EmbeddingBackend, EmbeddingBackendError
from .embedding_store import EmbeddingStore
from .models import EmbedReport
from .preprocessing import InvalidImageError, load_image

EmbedProgress = Callable[[int, int], None]  # (done, total_to_process)


def build_embeddings(
    db: Database,
    backend: EmbeddingBackend,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    purge_other_models: bool = False,
    on_progress: EmbedProgress | None = None,
) -> EmbedReport:
    """Embed every indexed image that lacks a current embedding for ``backend``.

    * Images that already have a current embedding are skipped.
    * Changed files (different SHA-256 than the stored vector) are re-embedded.
    * ``force`` re-embeds everything for this model (a rebuild).
    * Missing, corrupt, or modified-since-indexing files are skipped and
      reported; run ``index`` to refresh or prune them.
    * Work is committed batch by batch, so an interrupted run resumes where it
      stopped.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    store = EmbeddingStore(db)
    model_id = backend.model_id
    pending = store.pending(model_id, force=force)
    total_images = db.count()
    report = EmbedReport(
        model_id=model_id,
        total=total_images,
        up_to_date=total_images - len(pending),
    )

    done = 0
    for start in range(0, len(pending), batch_size):
        ids: list[int] = []
        shas: list[str] = []
        images = []
        for item in pending[start : start + batch_size]:
            path = Path(item.path)
            try:
                stat = path.stat()
            except OSError:
                report.skipped.append((item.path, "file not found (run 'index' to prune)"))
                continue
            if stat.st_size != item.file_size or stat.st_mtime_ns != item.mtime_ns:
                report.skipped.append(
                    (item.path, "changed since it was indexed (run 'index' first)")
                )
                continue
            try:
                loaded = load_image(path)
            except InvalidImageError as exc:
                report.skipped.append((item.path, str(exc)))
                continue
            ids.append(item.image_id)
            shas.append(item.sha256)
            images.append(loaded.image)

        if images:
            vectors = backend.embed_images(images)
            if vectors.shape[0] != len(images):
                raise EmbeddingBackendError(
                    f"backend returned {vectors.shape[0]} vectors for {len(images)} images"
                )
            store.upsert_many(model_id, zip(ids, shas, vectors))
            db.commit()
            report.embedded += len(images)
            for image in images:
                image.close()

        done += len(pending[start : start + batch_size])
        if on_progress is not None:
            on_progress(done, len(pending))

    if purge_other_models:
        report.purged = store.delete_other_models(model_id)
    db.commit()
    return report
