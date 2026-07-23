from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synth import (
    GalleryArtifact,
    GalleryManifest,
    publish_gallery_run,
    repository_relative_output_path,
    write_gallery_manifest,
)


def test_write_gallery_manifest_is_complete_and_portable(tmp_path: Path) -> None:
    repo_root = tmp_path
    files_directory = (
        repo_root / "data" / "outputs" / "example_v1_20260723_120000_000000" / "files"
    )
    files_directory.mkdir(parents=True)
    audio_path = files_directory / "example.wav"
    audio_path.write_bytes(b"RIFFexample")
    artifact = GalleryArtifact.from_file(repo_root, audio_path, "audio", "prediction")

    manifest_path = write_gallery_manifest(
        repo_root=repo_root,
        files_directory=files_directory,
        generator="example_v1",
        page_path="/example",
        artifacts=(artifact,),
    )

    manifest = GalleryManifest.model_validate_json(manifest_path.read_text())
    assert manifest.status == "complete"
    assert manifest.artifacts[0].path == Path(
        "data/outputs/example_v1_20260723_120000_000000/files/example.wav"
    )
    assert json.loads(manifest_path.read_text())["schema_version"] == 1


def test_gallery_artifact_rejects_absolute_paths() -> None:
    with pytest.raises(ValidationError):
        GalleryArtifact(
            path=Path("/tmp/audio.wav"),
            kind="audio",
            role="prediction",
            size_bytes=1,
            sha256="0" * 64,
        )


def test_publish_gallery_run_builds_transitive_artifacts(tmp_path: Path) -> None:
    files_directory = (
        tmp_path
        / "data"
        / "outputs"
        / "example_v1_20260723_120000_000000"
        / "files"
    )
    files_directory.mkdir(parents=True)
    report = files_directory / "report.json"
    audio = files_directory / "comparison.wav"
    report.write_text("{}\n", encoding="utf-8")
    audio.write_bytes(b"RIFFexample")

    manifest_path = publish_gallery_run(
        tmp_path,
        files_directory,
        "example_v1",
        "/example",
        (
            (report, "json", "report"),
            (audio, "audio", "comparison"),
        ),
    )

    manifest = GalleryManifest.model_validate_json(manifest_path.read_text())
    assert tuple(artifact.role for artifact in manifest.artifacts) == (
        "report",
        "comparison",
    )
    assert repository_relative_output_path(tmp_path, audio) == Path(
        "data/outputs/example_v1_20260723_120000_000000/files/comparison.wav"
    )
    assert repository_relative_output_path(
        tmp_path, files_directory / "future.wav"
    ) == Path(
        "data/outputs/example_v1_20260723_120000_000000/files/future.wav"
    )
