#!/usr/bin/env -S uv run

from __future__ import annotations

import re
import shlex
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from synth import GalleryManifest, file_sha256


CONFIG_PATH = Path(__file__).with_suffix(".toml")
HOST_PATTERN = re.compile(r"^[A-Za-z0-9_.@-]+$")


class RemoteConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = Field(pattern=HOST_PATTERN.pattern)
    repo: Path


class LocalConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: Path
    sync_webui: bool


class SelectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    latest_manifests: int = Field(gt=0)
    manifest_glob: str = Field(min_length=1)


class PullConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    remote: RemoteConfig
    local: LocalConfig
    selection: SelectionConfig


@dataclass(frozen=True)
class RemoteManifest:
    remote_path: Path
    manifest: GalleryManifest


def load_config(config_path: Path = CONFIG_PATH) -> PullConfig:
    config = PullConfig.model_validate(tomllib.loads(config_path.read_text(encoding="utf-8")))
    remote_repo = config.remote.repo
    if not remote_repo.is_absolute():
        raise ValueError("remote.repo must be absolute")
    local_repo = (config_path.parent / config.local.repo).resolve(strict=True)
    return config.model_copy(
        update={"local": config.local.model_copy(update={"repo": local_repo})}
    )


def ssh(config: RemoteConfig, arguments: tuple[str, ...]) -> str:
    command = shlex.join(arguments)
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", config.host, command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def remote_manifests(config: PullConfig) -> tuple[RemoteManifest, ...]:
    remote_pattern = config.remote.repo / config.selection.manifest_glob
    output = ssh(
        config.remote,
        ("find", str(config.remote.repo / "data" / "outputs"), "-path", str(remote_pattern), "-type", "f", "-print"),
    )
    manifests = []
    for path_text in output.splitlines():
        remote_path = Path(path_text)
        payload = ssh(config.remote, ("cat", str(remote_path)))
        manifests.append(
            RemoteManifest(
                remote_path=remote_path,
                manifest=GalleryManifest.model_validate_json(payload),
            )
        )
    return tuple(
        sorted(manifests, key=lambda item: item.manifest.created_at, reverse=True)[
            : config.selection.latest_manifests
        ]
    )


def relative_remote_path(config: PullConfig, path: Path) -> Path:
    return path.relative_to(config.remote.repo)


def pull_files(config: PullConfig, manifests: tuple[RemoteManifest, ...]) -> None:
    paths = {
        relative_remote_path(config, remote.remote_path)
        for remote in manifests
    }
    paths.update(
        artifact.path
        for remote in manifests
        for artifact in remote.manifest.artifacts
    )
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as file_list:
        file_list.write("\n".join(sorted(path.as_posix() for path in paths)) + "\n")
        file_list.flush()
        subprocess.run(
            [
                "rsync",
                "-a",
                "--partial",
                "--files-from",
                file_list.name,
                f"{config.remote.host}:{config.remote.repo.as_posix()}/",
                f"{config.local.repo.as_posix()}/",
            ],
            check=True,
        )


def sync_webui(config: PullConfig) -> None:
    if not config.local.sync_webui:
        return
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude",
            "__pycache__/",
            f"{config.remote.host}:{config.remote.repo.as_posix()}/webui/",
            f"{(config.local.repo / 'webui').as_posix()}/",
        ],
        check=True,
    )


def verify(config: PullConfig, manifests: tuple[RemoteManifest, ...]) -> None:
    for remote in manifests:
        for artifact in remote.manifest.artifacts:
            path = config.local.repo / artifact.path
            if path.stat().st_size != artifact.size_bytes:
                raise ValueError(f"Size mismatch for {artifact.path}")
            digest = file_sha256(path)
            if digest != artifact.sha256:
                raise ValueError(f"SHA-256 mismatch for {artifact.path}")


def main() -> None:
    config = load_config()
    manifests = remote_manifests(config)
    if not manifests:
        raise ValueError(
            f"No complete gallery manifests matched {config.selection.manifest_glob!r} "
            f"on {config.remote.host}"
        )
    pull_files(config, manifests)
    sync_webui(config)
    verify(config, manifests)
    for remote in manifests:
        print(
            f"{remote.manifest.created_at.isoformat()} "
            f"{remote.manifest.run_name} {remote.manifest.page_path}"
        )


if __name__ == "__main__":
    main()
