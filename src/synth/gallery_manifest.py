from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from synth.utils import file_sha256


GalleryArtifactKind = Literal["audio", "image", "html", "json", "preset", "text"]


class GalleryArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    kind: GalleryArtifactKind
    role: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: Path) -> Path:
        if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("data", "outputs"):
            raise ValueError("Artifact paths must be repository-relative under data/outputs")
        return path

    @classmethod
    def from_file(
        cls,
        repo_root: Path,
        path: Path,
        kind: GalleryArtifactKind,
        role: str,
    ) -> Self:
        resolved_root = repo_root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        relative_path = resolved_path.relative_to(resolved_root)
        return cls(
            path=relative_path,
            kind=kind,
            role=role,
            size_bytes=resolved_path.stat().st_size,
            sha256=file_sha256(resolved_path),
        )


class GalleryManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    status: Literal["complete"] = "complete"
    generator: str = Field(min_length=1)
    run_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    created_at: datetime
    page_path: str = Field(pattern=r"^/")
    artifacts: tuple[GalleryArtifact, ...] = Field(min_length=1)


def write_gallery_manifest(
    repo_root: Path,
    files_directory: Path,
    generator: str,
    page_path: str,
    artifacts: tuple[GalleryArtifact, ...],
) -> Path:
    resolved_root = repo_root.resolve(strict=True)
    resolved_files = files_directory.resolve(strict=True)
    resolved_files.relative_to(resolved_root / "data" / "outputs")
    if resolved_files.name != "files":
        raise ValueError(f"Expected a files directory, received {resolved_files}")

    manifest = GalleryManifest(
        generator=generator,
        run_name=resolved_files.parent.name,
        created_at=datetime.now().astimezone(),
        page_path=page_path,
        artifacts=artifacts,
    )
    destination = resolved_files / "gallery_manifest.json"
    temporary = resolved_files / ".gallery_manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
