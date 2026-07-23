#!/usr/bin/env -S uv run

from __future__ import annotations

import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Final

from synth.vital_runtime import sha256_bytes


LIBC_URL: Final[str] = (
    "https://deb.debian.org/debian/pool/main/g/glibc/"
    "libc6_2.41-12+deb13u4_amd64.deb"
)
LIBC_SHA256: Final[str] = (
    "967aa62605721081c3eb2a17650611a792aa802d76a6511d1840242623d204c9"
)
LIBSTDCXX_URL: Final[str] = (
    "https://deb.debian.org/debian/pool/main/g/gcc-14/"
    "libstdc++6_14.2.0-19_amd64.deb"
)
LIBSTDCXX_SHA256: Final[str] = (
    "ab1fa05837aa7a92aae748fd07a18a35f7d18bb4a71c4724fe2bbf0e32089de0"
)
VITAL_BINARY_SHA256: Final[str] = (
    "0eabf454a5dbfb2f9ce1f9496a98637094d585bcb6ca7bd32b21d524668adec5"
)


def download(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file():
        data = destination.read_bytes()
    else:
        with urllib.request.urlopen(url) as response:
            data = response.read()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    received = sha256_bytes(data)
    if received != expected_sha256:
        raise ValueError(
            f"Expected SHA-256 {expected_sha256} for {destination}, received {received}"
        )


def extract_package(package: Path, root: Path) -> None:
    subprocess.run(["dpkg-deb", "-x", str(package), str(root)], check=True)


def install_vital(installer: Path, destination: Path) -> None:
    member_root = "VitalInstaller/lib/vst3/Vital.vst3/"
    stage = destination.parent / ".Vital.vst3.stage"
    if stage.exists():
        shutil.rmtree(stage)
    with zipfile.ZipFile(installer) as archive:
        for member in archive.infolist():
            if member.filename.startswith(member_root):
                relative = Path(member.filename.removeprefix(member_root))
                if relative.parts:
                    output = stage / relative
                    if member.is_dir():
                        output.mkdir(parents=True, exist_ok=True)
                    else:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_bytes(archive.read(member))
    binary = stage / "Contents" / "x86_64-linux" / "Vital.so"
    if sha256_bytes(binary.read_bytes()) != VITAL_BINARY_SHA256:
        raise ValueError("The supplied Vital installer binary has changed")
    binary.chmod(0o755)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace installed Vital: {destination}")
    stage.rename(destination)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    local_root = repo_root / "local" / "vital-runtime"
    package_root = local_root / "packages"
    runtime_root = local_root / "root"
    packages = (
        (LIBC_URL, package_root / Path(LIBC_URL).name, LIBC_SHA256),
        (LIBSTDCXX_URL, package_root / Path(LIBSTDCXX_URL).name, LIBSTDCXX_SHA256),
    )
    for url, path, digest in packages:
        download(url, path, digest)
        extract_package(path, runtime_root)

    vital_destination = Path.home() / ".vst3" / "Vital.vst3"
    if not vital_destination.exists():
        vital_destination.parent.mkdir(parents=True, exist_ok=True)
        install_vital(Path.home() / "VitalInstaller.zip", vital_destination)
    print(vital_destination)


if __name__ == "__main__":
    main()
