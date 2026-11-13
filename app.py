#!/usr/bin/env python3
# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""SceneLift Studio launcher.

    python app.py                       # http://localhost:8000
    python app.py --port 9000           # another port
    python app.py --scenes-dir ~/scenes # another scene library
    python app.py --help                # every option

The browser UI is already compiled into scenelift/studio/static, so this one
command is all that is needed. No Node.js, no second process.
"""

import sys


def main(argv=None) -> int:
    if sys.version_info < (3, 10):
        print("SceneLift needs Python 3.10 or newer.", file=sys.stderr)
        return 1
    try:
        from scenelift.studio.server import main as studio_main
    except ImportError as e:
        print(f"A dependency is missing ({e}).\n"
              f"Install everything with:  pip install -r requirements.txt", file=sys.stderr)
        return 1
    return studio_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
