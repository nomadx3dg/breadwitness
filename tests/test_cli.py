import shutil

from typer.testing import CliRunner

from breadwitness import __version__
from breadwitness.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and __version__ in result.output


def test_index_stats_search_flow(image_dir, db_path, tmp_path):
    db = str(db_path)
    result = runner.invoke(app, ["index", str(image_dir), "--db", db])
    assert result.exit_code == 0, result.output
    assert "added:     3" in result.output and "skipped:   1" in result.output

    result = runner.invoke(app, ["stats", "--db", db])
    assert result.exit_code == 0 and "Images indexed:   3" in result.output

    query = tmp_path / "q.png"
    shutil.copy(image_dir / "a.png", query)
    result = runner.invoke(app, ["search", str(query), "--db", db, "--limit", "5"])
    assert result.exit_code == 0, result.output
    assert "a.png" in result.output and "exact" in result.output
    assert "not a probability" in result.output
    assert "pHash dist: 0/64" in result.output


def test_search_without_index_fails_cleanly(image_dir, tmp_path):
    result = runner.invoke(
        app, ["search", str(image_dir / "a.png"), "--db", str(tmp_path / "none.db")]
    )
    assert result.exit_code == 1 and "no index found" in result.output


def test_search_rejects_invalid_query(image_dir, db_path):
    runner.invoke(app, ["index", str(image_dir), "--db", str(db_path)])
    result = runner.invoke(
        app, ["search", str(image_dir / "corrupt.jpg"), "--db", str(db_path)]
    )
    assert result.exit_code == 1 and "not a valid image" in result.output


def test_duplicates_command(image_dir, db_path):
    shutil.copy(image_dir / "a.png", image_dir / "dup.png")
    runner.invoke(app, ["index", str(image_dir), "--db", str(db_path)])
    result = runner.invoke(app, ["duplicates", "--db", str(db_path)])
    assert result.exit_code == 0 and "2 files" in result.output


def test_models_command():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0 and "color-hist" in result.output and "openclip" in result.output


def test_search_auto_falls_back_to_hash_without_embeddings(image_dir, db_path):
    db = str(db_path)
    runner.invoke(app, ["index", str(image_dir), "--db", db])
    result = runner.invoke(app, ["search", str(image_dir / "a.png"), "--db", db, "--model", "color-hist"])
    assert result.exit_code == 0, result.output
    assert "using hash matching" in result.output and "Mode:" not in result.output


def test_embedding_flow(image_dir, db_path, tmp_path):
    db = str(db_path)
    runner.invoke(app, ["index", str(image_dir), "--db", db])

    result = runner.invoke(app, ["embed", "--db", db, "--model", "color-hist"])
    assert result.exit_code == 0, result.output
    assert "embedded:    3" in result.output
    result = runner.invoke(app, ["embed", "--db", db, "--model", "color-hist"])
    assert "embedded:    0" in result.output and "up to date:  3" in result.output
    result = runner.invoke(app, ["embed", "--db", db, "--model", "color-hist", "--rebuild"])
    assert "embedded:    3" in result.output

    result = runner.invoke(app, ["stats", "--db", db])
    assert "color-hist: 3/3 current" in result.output

    from .conftest import make_image

    query = tmp_path / "q.jpg"
    make_image(1).resize((128, 128)).save(query, quality=60)
    result = runner.invoke(app, ["search", str(query), "--db", db, "--model", "color-hist"])
    assert result.exit_code == 0, result.output
    assert "Mode: hybrid (model: color-hist)" in result.output
    assert "embedding:  cosine" in result.output and "basis: hash" in result.output

    result = runner.invoke(
        app, ["search", str(query), "--db", db, "--model", "color-hist",
              "--mode", "embedding", "--min-similarity", "-1"],
    )
    assert "semantic (embedding similarity)" in result.output


def test_embedding_search_without_embeddings_fails_cleanly(image_dir, db_path):
    db = str(db_path)
    runner.invoke(app, ["index", str(image_dir), "--db", db])
    result = runner.invoke(
        app, ["search", str(image_dir / "a.png"), "--db", db,
              "--mode", "hybrid", "--model", "color-hist"],
    )
    assert result.exit_code == 1 and "no embeddings" in result.output


def test_index_with_embed_flag(image_dir, db_path):
    result = runner.invoke(
        app, ["index", str(image_dir), "--db", str(db_path), "--embed", "--model", "color-hist"]
    )
    assert result.exit_code == 0, result.output
    assert "added:     3" in result.output and "embedded:    3" in result.output


def test_bad_model_fails_before_indexing(image_dir, db_path):
    result = runner.invoke(
        app, ["index", str(image_dir), "--db", str(db_path), "--embed", "--model", "nope"]
    )
    assert result.exit_code == 1 and "Unknown model" in result.output
