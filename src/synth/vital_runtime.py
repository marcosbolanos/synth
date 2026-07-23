from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class VitalRuntime(BaseModel):
    """Paths required to run a Python Vital worker with a user-local glibc."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    loader: Path
    library_directory: Path
    python: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> VitalRuntime:
        runtime_library = (
            repo_root
            / "local"
            / "vital-runtime"
            / "root"
            / "usr"
            / "lib"
            / "x86_64-linux-gnu"
        )
        return cls(
            loader=(runtime_library / "ld-linux-x86-64.so.2").resolve(strict=True),
            library_directory=runtime_library.resolve(strict=True),
            python=Path(sys.executable).absolute(),
        )

    def command(self, script: Path, *arguments: str) -> tuple[str, ...]:
        host_libraries = (
            Path("/lib/x86_64-linux-gnu"),
            Path("/usr/lib/x86_64-linux-gnu"),
        )
        library_path = os.pathsep.join(
            str(path) for path in (self.library_directory, *host_libraries)
        )
        return (
            str(self.loader),
            "--library-path",
            library_path,
            str(self.python),
            str(script.resolve(strict=True)),
            *arguments,
        )

    def run_worker(
        self,
        script: Path,
        *arguments: str,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.command(script, *arguments),
            check=True,
            capture_output=capture_output,
            text=True,
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
