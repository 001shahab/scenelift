# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""SceneLift Studio: a local web app on top of the scenelift library.

Studio reconstructs scenes (from uploads, or from the built-in synthetic room
that needs no GPU), shows them in 3D, and exposes every edit, spatial query and
selection of the library through a small REST API and a React front end.

    python app.py                 # from a checkout
    scenelift studio              # after `pip install -e .`

Nothing here adds to the method. Every number shown in the browser is computed
by ``scenelift.query`` on the reconstruction; see docs/DESIGN.md.
"""

from .app import create_app
from .server import serve
from .settings import StudioSettings

__all__ = ["StudioSettings", "create_app", "serve"]
