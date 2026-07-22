#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import os
import re
from email.message import Message
from pathlib import Path

import httpx


PRESETSHARE_ORIGIN = "https://presetshare.com"
CODE_PATTERN = re.compile(r"p[1-9][0-9]*")
URL_PATTERN = re.compile(
    r"https?://(?:www\.)?presetshare\.com/(?P<code>p[1-9][0-9]*)/?"
)


def parse_code(source: str) -> str:
    """Return a canonical PresetShare code from a code or preset URL."""
    if CODE_PATTERN.fullmatch(source):
        return source

    url_match = URL_PATTERN.fullmatch(source)
    if url_match is not None:
        return url_match.group("code")

    raise argparse.ArgumentTypeError(
        "expected a code such as p22689 or a URL such as "
        "https://presetshare.com/p22689"
    )


def response_filename(response: httpx.Response) -> str:
    """Extract the server-provided preset filename."""
    content_disposition = response.headers["Content-Disposition"]
    message = Message()
    message["Content-Disposition"] = content_disposition
    filename = message.get_filename()
    if filename is None:
        raise RuntimeError("PresetShare did not provide a download filename")
    return Path(filename).name


def download_preset(repo_root: Path, code: str, cookie: str) -> Path:
    """Download one authenticated PresetShare preset into the external data tree."""
    preset_id = code.removeprefix("p")
    download_url = f"{PRESETSHARE_ORIGIN}/download?id={preset_id}"
    response = httpx.get(
        download_url,
        headers={
            "Cookie": cookie,
            "User-Agent": "synth-experiments-preset-downloader/1.0",
        },
        follow_redirects=True,
        timeout=30.0,
    )
    response.raise_for_status()

    if response.url.path == "/user/login":
        raise RuntimeError(
            "PresetShare rejected PRESETSHARE_COOKIE; copy a fresh Cookie request "
            "header from an authenticated browser session"
        )

    output_dir = repo_root / "data/external/presets/presetshare" / code
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / response_filename(response)
    output_file.write_bytes(response.content)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a preset using an authenticated PresetShare session."
    )
    parser.add_argument(
        "source",
        type=parse_code,
        help="PresetShare code or full preset URL",
    )
    args = parser.parse_args()

    cookie = os.environ.get("PRESETSHARE_COOKIE")
    if cookie is None:
        parser.error(
            "PRESETSHARE_COOKIE must contain the Cookie request header from an "
            "authenticated PresetShare browser session"
        )

    repo_root = Path(__file__).resolve().parent.parent
    output_file = download_preset(repo_root, args.source, cookie)
    print(output_file)


if __name__ == "__main__":
    main()
