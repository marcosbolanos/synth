#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "data" / "outputs"
app = FastAPI(title="Synth output gallery", docs_url=None, redoc_url=None)


@dataclass(frozen=True)
class OutputFile:
    relative_path: Path
    size_bytes: int
    modified_at: datetime

    @property
    def url(self) -> str:
        return f"/files/{quote(self.relative_path.as_posix())}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve generated outputs to a tailnet.")
    parser.add_argument(
        "--host",
        required=True,
        help="Exact Tailscale IP on which to listen; do not use 0.0.0.0.",
    )
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def output_files() -> tuple[OutputFile, ...]:
    files = []
    for path in OUTPUT_ROOT.rglob("*"):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        stat = path.stat()
        files.append(
            OutputFile(
                relative_path=path.relative_to(OUTPUT_ROOT),
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime).astimezone(),
            )
        )
    return tuple(sorted(files, key=lambda item: item.modified_at, reverse=True))


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    if size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.1f} KB"
    return f"{size_bytes} B"


def file_card(item: OutputFile) -> str:
    label = html.escape(item.relative_path.as_posix())
    details = (
        f"{format_size(item.size_bytes)} · "
        f"{item.modified_at.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )
    suffix = item.relative_path.suffix.lower()
    preview = ""
    if suffix in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        preview = f'<audio controls preload="metadata" src="{item.url}"></audio>'
    elif suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
        preview = f'<a href="{item.url}"><img loading="lazy" src="{item.url}"></a>'
    elif suffix == ".html":
        preview = f'<a class="button" href="{item.url}">Open HTML</a>'

    return f"""
      <article>
        <a class="filename" href="{item.url}">{label}</a>
        <div class="details">{details}</div>
        {preview}
      </article>
    """


def page(title: str, cards: str, active_path: str) -> str:
    outputs_class = ' class="active"' if active_path == "/" else ""
    midtempo_class = ' class="active"' if active_path == "/midtempo" else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synth outputs</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 52rem; margin: auto; padding: 1rem; background: #111318; }}
    header {{ display: flex; align-items: center; justify-content: space-between; }}
    h1 {{ font-size: 1.4rem; }}
    nav {{ display: flex; gap: .5rem; margin: .4rem 0 1rem; }}
    nav a {{ border: 1px solid #3b4452; border-radius: 999px; padding: .45rem .8rem;
             text-decoration: none; }}
    nav a.active {{ color: white; background: #315c91; border-color: #315c91; }}
    article {{ background: #1c2028; border: 1px solid #303642; border-radius: .7rem;
               margin: .8rem 0; padding: .9rem; overflow-wrap: anywhere; }}
    a {{ color: #8bc5ff; }}
    .filename {{ font-weight: 650; }}
    .details {{ color: #aab2c0; font-size: .82rem; margin: .35rem 0 .7rem; }}
    .comments {{ color: #c8ced8; font-size: .9rem; white-space: pre-line; }}
    .badge {{ display: inline-block; background: #303642; border-radius: 999px;
              font-size: .75rem; margin-left: .4rem; padding: .15rem .45rem; }}
    audio {{ display: block; width: 100%; }}
    img {{ display: block; max-width: 100%; max-height: 32rem; border-radius: .4rem; }}
    .button, button {{ display: inline-block; color: white; background: #315c91;
                       border: 0; border-radius: .4rem; padding: .5rem .7rem;
                       text-decoration: none; font: inherit; }}
  </style>
</head>
<body>
  <header><h1>{html.escape(title)}</h1><button onclick="location.reload()">Refresh</button></header>
  <nav><a href="/"{outputs_class}>All outputs</a><a href="/midtempo"{midtempo_class}>Midtempo</a></nav>
  {cards}
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def gallery() -> str:
    cards = "\n".join(file_card(item) for item in output_files())
    return page("Synth outputs", cards or "<p>No generated outputs yet.</p>", "/")


def latest_midtempo_manifest() -> Path | None:
    manifests = tuple(OUTPUT_ROOT.glob("render_midtempo_vital_v1_*/files/midtempo_manifest.json"))
    return max(manifests, key=lambda path: path.stat().st_mtime) if manifests else None


@app.get("/midtempo", response_class=HTMLResponse)
def midtempo_gallery() -> str:
    manifest_path = latest_midtempo_manifest()
    if manifest_path is None:
        return page("Midtempo presets", "<p>No Midtempo render exists yet.</p>", "/midtempo")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cards = []
    for preset in manifest["presets"]:
        audio_path = manifest_path.parent / preset["audio_file"]
        audio_url = f"/files/{quote(audio_path.relative_to(OUTPUT_ROOT).as_posix())}"
        name = html.escape(preset["preset_name"])
        style = html.escape(preset["preset_style"])
        author = html.escape(preset["author"])
        comments = html.escape(preset["comments"])
        cards.append(
            f"""
            <article>
              <div class="filename">{preset['index']:02d}. {name}<span class="badge">{style}</span></div>
              <div class="details">{author} · sustained C2 · 6 seconds</div>
              <audio controls preload="metadata" src="{audio_url}"></audio>
              <p class="comments">{comments}</p>
            </article>
            """
        )
    return page("BLA Midtempo for Vital", "\n".join(cards), "/midtempo")


@app.get("/files/{relative_path:path}", response_class=FileResponse)
def generated_file(relative_path: str) -> Path:
    requested = (OUTPUT_ROOT / relative_path).resolve()
    if not requested.is_relative_to(OUTPUT_ROOT.resolve()) or not requested.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")
    return requested


def main() -> None:
    args = parse_args()
    if args.host == "0.0.0.0":
        raise ValueError("Refusing to expose the output gallery on every interface")
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
