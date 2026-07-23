#!/usr/bin/env -S uv run

from __future__ import annotations

from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Final

import httpx
import py7zr
from synth import file_sha256


SOURCE_URL: Final[str] = (
    "https://www.dropbox.com/scl/fi/9okgsmcnmtdyxl9hpcjy3/"
    "Jek-s-Vital-Presets.7z?rlkey=haxivanm4761nxq5qjjsrctwj&dl=1"
)
EXPECTED_ARCHIVE_BYTES: Final[int] = 1_326_243_416
ARCHIVE_NAME: Final[str] = "Jeks-Vital-Presets.7z"


def download_archive(destination: Path, partial: Path) -> None:
    received_bytes = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={received_bytes}-"} if received_bytes else {}
    transport = httpx.HTTPTransport(local_address="0.0.0.0")
    with httpx.Client(
        transport=transport,
        follow_redirects=True,
        timeout=None,
    ) as client:
        with client.stream("GET", SOURCE_URL, headers=headers) as response:
            response.raise_for_status()
            if received_bytes and response.status_code != httpx.codes.PARTIAL_CONTENT:
                raise ValueError("Preset archive server did not honor the resume range")
            mode = "ab" if received_bytes else "wb"
            with partial.open(mode) as output:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    output.write(chunk)

    received_bytes = partial.stat().st_size
    if received_bytes != EXPECTED_ARCHIVE_BYTES:
        raise ValueError(
            f"Expected {EXPECTED_ARCHIVE_BYTES} archive bytes, received {received_bytes}"
        )
    partial.replace(destination)


def validate_member_names(names: list[str]) -> None:
    for name in names:
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe archive member path: {name}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    category_root = repo_root / "data" / "external" / "Vital Presets"
    pack_root = category_root / "Jeks Vital Presets"
    partial_archive = category_root / f".{ARCHIVE_NAME}.partial"

    if pack_root.exists():
        raise FileExistsError(
            f"Refusing to replace an existing preset pack directory: {pack_root}"
        )

    category_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="jeks_vital_presets_", dir=category_root) as temporary:
        staged_pack = Path(temporary)
        archive_dir = staged_pack / "archive"
        files_dir = staged_pack / "files"
        archive_path = archive_dir / ARCHIVE_NAME
        archive_dir.mkdir()
        download_archive(archive_path, partial_archive)

        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            names = archive.getnames()
            validate_member_names(names)
            files_dir.mkdir()
            archive.extractall(path=files_dir)

        preset_count = sum(1 for path in files_dir.rglob("*.vital"))
        bank_count = sum(1 for path in files_dir.rglob("*.vitalbank"))
        extracted_bytes = sum(
            path.stat().st_size for path in files_dir.rglob("*") if path.is_file()
        )
        archive_digest = file_sha256(archive_path)
        staged_pack.rename(pack_root)

    print(f"pack={pack_root}")
    print(f"archive_sha256={archive_digest}")
    print(f"vital_presets={preset_count}")
    print(f"vital_banks={bank_count}")
    print(f"extracted_bytes={extracted_bytes}")


if __name__ == "__main__":
    main()
