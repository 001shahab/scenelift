# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Start the Studio server. Used by `python app.py`, `scenelift studio` and
`python -m scenelift.studio`."""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser
from typing import Optional

from .. import __version__
from .settings import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_SCENES_DIR, StudioSettings


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--host", default=None,
                    help=f"interface to bind (default {DEFAULT_HOST}; use 0.0.0.0 to serve your network)")
    ap.add_argument("--port", type=int, default=None, help=f"port (default {DEFAULT_PORT})")
    ap.add_argument("--scenes-dir", default=None,
                    help=f"folder that holds the scenes (default ./{DEFAULT_SCENES_DIR})")
    ap.add_argument("--device", default=None,
                    help="torch device for the model backends, e.g. cuda, mps, cpu (default: automatic)")
    ap.add_argument("--sam3d-repo", default=None,
                    help="path to a sam-3d-objects checkout; enables generative completion")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")


def settings_from_args(a: argparse.Namespace) -> StudioSettings:
    """Command-line flags win over environment variables, which win over defaults."""
    settings = StudioSettings.from_env(scenes_dir=a.scenes_dir, host=a.host, port=a.port,
                                       device=a.device, sam3d_repo=a.sam3d_repo)
    settings.open_browser = not a.no_browser
    return settings


def serve(settings: Optional[StudioSettings] = None) -> None:
    """Run the server until interrupted."""
    import uvicorn

    from .app import STATIC_DIR, create_app

    settings = settings or StudioSettings.from_env()
    app = create_app(settings)
    shown_host = "localhost" if settings.host in ("127.0.0.1", "0.0.0.0", "::") else settings.host
    url = f"http://{shown_host}:{settings.port}"
    print(f"\n  SceneLift Studio {__version__}  |  Film it. Edit it in 3D.\n"
          f"  Copyright 2026 3S Holding OÜ. Apache-2.0.\n\n"
          f"  Open      {url}\n"
          f"  API docs  {url}/api/docs\n"
          f"  Scenes    {settings.scenes_dir.resolve()}\n")
    if not (STATIC_DIR / "index.html").is_file():
        print("  Note: the compiled front end is missing. Build it with\n"
              "        cd frontend && npm install && npm run build\n")
    if settings.open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    logging.getLogger("scenelift.studio").setLevel(logging.INFO)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="scenelift studio",
        description="SceneLift Studio: reconstruct, inspect, edit and query 3D scenes in the browser.")
    add_arguments(ap)
    ap.add_argument("--version", action="version", version=f"scenelift {__version__}")
    serve(settings_from_args(ap.parse_args(argv)))
    return 0
