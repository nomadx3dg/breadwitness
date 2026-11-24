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
