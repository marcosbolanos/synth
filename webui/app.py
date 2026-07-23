from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "data" / "outputs"
RUN_TIMESTAMP = re.compile(r"_(?P<timestamp>\d{8}_\d{6}(?:_\d{6})?)$")
AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"})

app = FastAPI(title="Synth output gallery", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=WEB_ROOT / "templates")


@dataclass(frozen=True)
class OutputFile:
    path: Path
    size_bytes: int
    modified_at: datetime

    @property
    def relative_path(self) -> Path:
        return self.path.relative_to(OUTPUT_ROOT)

    @property
    def url(self) -> str:
        return f"/files/{quote(self.relative_path.as_posix())}"

    @property
    def kind(self) -> str:
        suffix = self.path.suffix.lower()
        if suffix in AUDIO_SUFFIXES:
            return "audio"
        if suffix in IMAGE_SUFFIXES:
            return "image"
        if suffix == ".html":
            return "html"
        return "file"

    @property
    def display_size(self) -> str:
        if self.size_bytes >= 1_000_000:
            return f"{self.size_bytes / 1_000_000:.1f} MB"
        if self.size_bytes >= 1_000:
            return f"{self.size_bytes / 1_000:.1f} KB"
        return f"{self.size_bytes} B"


@dataclass(frozen=True)
class OutputRun:
    path: Path
    created_at: datetime

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def title(self) -> str:
        match = RUN_TIMESTAMP.search(self.name)
        stem = self.name[: match.start()] if match else self.name
        return stem.replace("_", " ").title()

    @property
    def is_comparison(self) -> bool:
        return (
            (self.path / "files" / "original").is_dir()
            and (self.path / "files" / "reconstructed").is_dir()
        )

    @property
    def url(self) -> str:
        prefix = "/dac" if self.is_comparison else "/runs"
        return f"{prefix}/{quote(self.name)}"

    @property
    def files(self) -> tuple[OutputFile, ...]:
        items = tuple(output_file(path) for path in self.path.rglob("*") if path.is_file())
        return tuple(sorted(items, key=lambda item: item.relative_path.as_posix()))


@dataclass(frozen=True)
class AudioComparison:
    name: str
    original: OutputFile
    reconstructed: OutputFile


def output_file(path: Path) -> OutputFile:
    stat = path.stat()
    return OutputFile(
        path=path,
        size_bytes=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime).astimezone(),
    )


def output_runs() -> tuple[OutputRun, ...]:
    runs = []
    for path in OUTPUT_ROOT.iterdir():
        if not path.is_dir():
            continue
        match = RUN_TIMESTAMP.search(path.name)
        created_at = (
            datetime.strptime(match.group("timestamp")[:15], "%Y%m%d_%H%M%S").astimezone()
            if match
            else datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        )
        runs.append(OutputRun(path=path, created_at=created_at))
    return tuple(sorted(runs, key=lambda run: run.created_at, reverse=True))


def output_run(run_name: str) -> OutputRun:
    requested = (OUTPUT_ROOT / run_name).resolve()
    if not requested.is_relative_to(OUTPUT_ROOT.resolve()) or not requested.is_dir():
        raise HTTPException(status_code=404, detail="Output run not found")
    return next(run for run in output_runs() if run.path.resolve() == requested)


def comparisons_for(run: OutputRun) -> tuple[AudioComparison, ...]:
    original_root = run.path / "files" / "original"
    reconstructed_root = run.path / "files" / "reconstructed"
    comparisons = []
    for original_path in sorted(original_root.rglob("*.wav")):
        relative_path = original_path.relative_to(original_root)
        reconstructed_path = reconstructed_root / relative_path
        if reconstructed_path.is_file():
            comparisons.append(
                AudioComparison(
                    name=relative_path.stem.removesuffix("_original").replace("_", " ").title(),
                    original=output_file(original_path),
                    reconstructed=output_file(reconstructed_path),
                )
            )
    return tuple(comparisons)


def page_context(request: Request, active_path: str, **values: object) -> dict[str, object]:
    return {"request": request, "active_path": active_path, "runs": output_runs(), **values}


@app.get("/", response_class=HTMLResponse)
def gallery(request: Request) -> HTMLResponse:
    runs = output_runs()
    if not runs:
        return templates.TemplateResponse(
            request=request,
            name="empty.html",
            context=page_context(request, "/", title="Synth outputs", message="No generated outputs yet."),
        )
    latest = runs[0]
    return templates.TemplateResponse(
        request=request,
        name="run.html",
        context=page_context(request, latest.url, title=f"Latest · {latest.title}", run=latest),
    )


@app.get("/runs/{run_name}", response_class=HTMLResponse)
def run_gallery(request: Request, run_name: str) -> HTMLResponse:
    run = output_run(run_name)
    return templates.TemplateResponse(
        request=request,
        name="run.html",
        context=page_context(request, run.url, title=run.title, run=run),
    )


@app.get("/dac", response_class=HTMLResponse)
def latest_dac_gallery(request: Request) -> HTMLResponse:
    run = next((candidate for candidate in output_runs() if candidate.is_comparison), None)
    if run is None:
        return templates.TemplateResponse(
            request=request,
            name="empty.html",
            context=page_context(request, "/dac", title="DAC comparison", message="No DAC comparison exists yet."),
        )
    return dac_gallery(request, run.name, active_path="/dac")


@app.get("/dac/{run_name}", response_class=HTMLResponse)
def dac_gallery(request: Request, run_name: str, active_path: str | None = None) -> HTMLResponse:
    run = output_run(run_name)
    if not run.is_comparison:
        raise HTTPException(status_code=404, detail="DAC comparison not found")
    return templates.TemplateResponse(
        request=request,
        name="comparison.html",
        context=page_context(
            request,
            active_path or run.url,
            title="DAC · Original vs reconstructed",
            run=run,
            comparisons=comparisons_for(run),
        ),
    )


@app.get("/midtempo", response_class=HTMLResponse)
def midtempo_gallery(request: Request) -> HTMLResponse:
    manifests = tuple(OUTPUT_ROOT.glob("render_midtempo_vital_v1_*/files/midtempo_manifest.json"))
    if not manifests:
        return templates.TemplateResponse(
            request=request,
            name="empty.html",
            context=page_context(request, "/midtempo", title="Midtempo presets", message="No Midtempo render exists yet."),
        )
    manifest_path = max(manifests, key=lambda path: path.stat().st_mtime)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    presets = tuple(
        {
            **preset,
            "audio_url": f"/files/{quote((manifest_path.parent / preset['audio_file']).relative_to(OUTPUT_ROOT).as_posix())}",
        }
        for preset in manifest["presets"]
    )
    return templates.TemplateResponse(
        request=request,
        name="midtempo.html",
        context=page_context(request, "/midtempo", title="BLA Midtempo for Vital", presets=presets),
    )


def latest_manifest(pattern: str) -> Path | None:
    manifests = tuple(OUTPUT_ROOT.glob(pattern))
    return max(manifests, key=lambda path: path.stat().st_mtime) if manifests else None


@app.get("/vital-investigation", response_class=HTMLResponse)
def vital_investigation(request: Request) -> HTMLResponse:
    manifest_path = latest_manifest(
        "investigate_vital_architecture_v1_*/files/architecture_investigation.json"
    )
    if manifest_path is None:
        return templates.TemplateResponse(
            request=request,
            name="empty.html",
            context=page_context(
                request,
                "/vital-investigation",
                title="Vital architecture investigation",
                message="No completed architecture investigation exists yet.",
            ),
        )
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    plot = output_file(manifest_path.parent / "audio_encoding_comparison.png")
    return templates.TemplateResponse(
        request=request,
        name="vital_investigation.html",
        context=page_context(
            request,
            "/vital-investigation",
            title="Vital architecture investigation",
            report=report,
            plot=plot,
        ),
    )


@app.get("/vital-augmentation", response_class=HTMLResponse)
def vital_augmentation(request: Request) -> HTMLResponse:
    manifest_path = latest_manifest(
        "compare_vital_augmentation_v1_*/files/augmentation_demo.json"
    )
    if manifest_path is None:
        return templates.TemplateResponse(
            request=request,
            name="empty.html",
            context=page_context(
                request,
                "/vital-augmentation",
                title="Vital augmentation chapter",
                message="No completed Vital augmentation comparison exists yet.",
            ),
        )
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparisons = json.loads(
        (manifest_path.parent / report["comparisons"]).read_text(encoding="utf-8")
    )
    prepared = tuple(
        {
            **item,
            "input_url": output_file(Path(item["input_audio"])).url,
            "baseline_output_url": output_file(Path(item["baseline_output_audio"])).url,
            "augmented_output_url": output_file(Path(item["augmented_output_audio"])).url,
            "baseline_preset_url": output_file(Path(item["baseline_preset"])).url,
            "augmented_preset_url": output_file(Path(item["augmented_preset"])).url,
        }
        for item in comparisons
    )
    plots = tuple(output_file(manifest_path.parent / name) for name in report["plots"])
    return templates.TemplateResponse(
        request=request,
        name="vital_augmentation.html",
        context=page_context(
            request,
            "/vital-augmentation",
            title="Vital augmentation · baseline vs four variants",
            report=report,
            plots=plots,
            train_comparisons=tuple(item for item in prepared if item["split"] != "test"),
            test_comparisons=tuple(item for item in prepared if item["split"] == "test"),
        ),
    )


@app.get("/vital-stream-scale", response_class=HTMLResponse)
def vital_stream_scale(request: Request) -> HTMLResponse:
    manifest_path = latest_manifest(
        "compare_vital_stream_scale_v1_*/files/stream_scale_demo.json"
    )
    if manifest_path is None:
        return templates.TemplateResponse(
            request=request,
            name="empty.html",
            context=page_context(
                request,
                "/vital-stream-scale",
                title="Vital streaming scale chapter",
                message="The diverse streaming augmentation experiment is still running.",
            ),
        )
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparisons = json.loads(
        (manifest_path.parent / report["comparisons"]).read_text(encoding="utf-8")
    )
    prepared = tuple(
        {
            **item,
            "input_url": output_file(Path(item["input_audio"])).url,
            **{
                f"{name}_output_url": output_file(
                    Path(item[f"{name}_output_audio"])
                ).url
                for name in ("baseline", "four_variant", "stream")
            },
            **{
                f"{name}_preset_url": output_file(
                    Path(item[f"{name}_preset"])
                ).url
                for name in ("baseline", "four_variant", "stream")
            },
        }
        for item in comparisons
    )
    plots = tuple(output_file(manifest_path.parent / name) for name in report["plots"])
    return templates.TemplateResponse(
        request=request,
        name="vital_stream_scale.html",
        context=page_context(
            request,
            "/vital-stream-scale",
            title="Vital streaming scale · diversified synthetic pool",
            report=report,
            plots=plots,
            train_comparisons=tuple(item for item in prepared if item["split"] != "test"),
            test_comparisons=tuple(item for item in prepared if item["split"] == "test"),
        ),
    )


def training_report_path(run_name: str | None = None) -> Path | None:
    if run_name is None:
        return latest_manifest("train_vital_transformer_v1_*/files/training_report.json")
    run = output_run(run_name)
    candidate = run.path / "files" / "training_report.json"
    return candidate if candidate.is_file() else None


def vital_training_page(
    request: Request,
    manifest_path: Path,
    active_path: str,
) -> HTMLResponse:
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    if report["status"] != "complete":
        return templates.TemplateResponse(
            request=request,
            name="empty.html",
            context=page_context(
                request,
                active_path,
                title="Vital sound-to-preset transformer",
                message=f"Training run status: {report['status'].replace('_', ' ')}.",
            ),
        )
    comparisons = json.loads(
        (manifest_path.parent / report["comparisons"]).read_text(encoding="utf-8")
    )
    prepared = []
    for comparison in comparisons:
        prepared.append(
            {
                **comparison,
                "input_url": output_file(manifest_path.parent / comparison["input_audio"]).url,
                "output_url": output_file(manifest_path.parent / comparison["output_audio"]).url,
                "preset_url": output_file(
                    manifest_path.parent / comparison["predicted_preset"]
                ).url,
            }
        )
    plots = tuple(output_file(manifest_path.parent / name) for name in report["plots"])
    return templates.TemplateResponse(
        request=request,
        name="vital_training.html",
        context=page_context(
            request,
            active_path,
            title="Vital sound-to-preset transformer",
            report=report,
            plots=plots,
            train_comparisons=tuple(item for item in prepared if item["split"] != "test"),
            test_comparisons=tuple(item for item in prepared if item["split"] == "test"),
        ),
    )


@app.get("/vital-transformer", response_class=HTMLResponse)
def latest_vital_training(request: Request) -> HTMLResponse:
    manifest_path = training_report_path()
    if manifest_path is None:
        return templates.TemplateResponse(
            request=request,
            name="empty.html",
            context=page_context(
                request,
                "/vital-transformer",
                title="Vital sound-to-preset transformer",
                message="No completed Vital transformer training run exists yet.",
            ),
        )
    return vital_training_page(request, manifest_path, "/vital-transformer")


@app.get("/vital-transformer/{run_name}", response_class=HTMLResponse)
def historical_vital_training(request: Request, run_name: str) -> HTMLResponse:
    manifest_path = training_report_path(run_name)
    if manifest_path is None:
        raise HTTPException(status_code=404, detail="Vital training report not found")
    return vital_training_page(
        request,
        manifest_path,
        f"/vital-transformer/{quote(run_name)}",
    )


@app.get("/files/{relative_path:path}", response_class=FileResponse)
def generated_file(relative_path: str) -> Path:
    requested = (OUTPUT_ROOT / relative_path).resolve()
    if not requested.is_relative_to(OUTPUT_ROOT.resolve()) or not requested.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve generated outputs to a tailnet.")
    parser.add_argument("--host", required=True, help="Exact Tailscale IP; never use 0.0.0.0.")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host == "0.0.0.0":
        raise ValueError("Refusing to expose the output gallery on every interface")
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
