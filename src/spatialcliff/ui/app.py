"""FastAPI app for the SpatialCliff console.

Serves the sweep analysis JSON (per-family complexity curves and falloff
locations) and the scene corpus manifest, plus a static dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from spatialcliff import __version__
from spatialcliff.analysis import SweepResult
from spatialcliff.ui import repo_root

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _read_json(root: Path, rel: str) -> Any:
    path = root / rel
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{rel} missing")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def create_app() -> FastAPI:
    app = FastAPI(title="SpatialCliff", version=__version__)
    root = repo_root()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "service": "spatialcliff"}

    @app.get("/api/spatialcliff/sweep.json")
    def sweep() -> Any:
        return _read_json(root, "data/sweep/sweep.json")

    @app.get("/api/spatialcliff/decay.json")
    def decay() -> dict[str, Any]:
        sweep_path = root / "data" / "sweep" / "sweep.json"
        if not sweep_path.is_file():
            raise HTTPException(status_code=404, detail="sweep results missing — run scripts/run_sweep.py")
        result = SweepResult.load(sweep_path)
        return {
            "summary": result.summary(),
            "families": sorted(result.curves),
            "curves": {
                fam: {
                    "labels": c.labels,
                    "complexity": c.complexity,
                    "acc": c.acc,
                    "ci_lo": c.ci_lo,
                    "ci_hi": c.ci_hi,
                    "falloff": c.falloff(),
                }
                for fam, c in result.curves.items()
            },
        }

    @app.get("/api/spatialcliff/manifest.json")
    def manifest() -> Any:
        return _read_json(root, "data/scenes/manifest.json")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        index_path = STATIC_DIR / "spatialcliff.html"
        if not index_path.is_file():
            raise HTTPException(status_code=500, detail="SpatialCliff UI static file missing")
        return FileResponse(index_path)

    return app
