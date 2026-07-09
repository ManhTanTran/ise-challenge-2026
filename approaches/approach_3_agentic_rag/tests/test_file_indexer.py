import json
from pathlib import Path

from approaches.approach_3_agentic_rag.shared_src import file_indexer as file_indexer_module


def _write_manifest(path: Path, items: list[dict]) -> None:
    path.write_text(json.dumps(items), encoding="utf-8")


def test_reindex_failed_files_only_retries_failing_status(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "lake"
    data_root.mkdir()
    (data_root / "good.txt").write_text("fine", encoding="utf-8")
    (data_root / "bad.csv").write_text("garbled", encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {"relative_path": "good.txt", "status": "ok", "error_message": None},
            {"relative_path": "bad.csv", "status": "error", "error_message": "ParserError"},
        ],
    )

    retried_paths = []

    def fake_index_file(path, data_lake_dir, *, cache_dir, force=False):
        retried_paths.append(Path(path).name)
        return {"relative_path": "bad.csv", "status": "ok", "error_message": None}

    monkeypatch.setattr(file_indexer_module, "index_file", fake_index_file)

    result = file_indexer_module.reindex_failed_files(manifest_path, data_root)

    assert retried_paths == ["bad.csv"]  # good.txt was never re-read
    by_path = {item["relative_path"]: item for item in result}
    assert by_path["good.txt"]["status"] == "ok"
    assert by_path["bad.csv"]["status"] == "ok"

    # Result is also persisted back to the manifest file.
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {item["relative_path"]: item["status"] for item in saved} == {
        "good.txt": "ok",
        "bad.csv": "ok",
    }


def test_reindex_failed_files_noop_when_nothing_failing(tmp_path: Path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, [{"relative_path": "good.txt", "status": "ok"}])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("index_file should not be called when nothing failed")

    monkeypatch.setattr(file_indexer_module, "index_file", fail_if_called)

    result = file_indexer_module.reindex_failed_files(manifest_path, tmp_path)
    assert result[0]["status"] == "ok"


def test_reindex_failed_files_skips_missing_source_file(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "lake"
    data_root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [{"relative_path": "gone.csv", "status": "error", "error_message": "boom"}],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("index_file should not be called for a missing source file")

    monkeypatch.setattr(file_indexer_module, "index_file", fail_if_called)

    result = file_indexer_module.reindex_failed_files(manifest_path, data_root)
    assert result[0]["status"] == "error"  # left untouched, not crashed
