# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Command line interface.

  scenelift run room.mp4 -o room_scene --export room.glb
  scenelift info room_scene
  scenelift measure room_scene sofa_1 table_1
  scenelift size room_scene sofa_1
  scenelift fit room_scene sofa_1 --space 2.2 1.0 0.9
  scenelift fit room_scene chair_1 --between table_1 cabinet_1
  scenelift place room_scene chair_1
  scenelift edit room_scene remove table_1
  scenelift edit room_scene move sofa_1 0.5 0 0
  scenelift edit room_scene rotate sofa_1 90
  scenelift calibrate room_scene door_1 height 2.03
  scenelift export room_scene room.glb
  scenelift eval room_scene ground_truth.json
  scenelift studio                      (the browser UI; same as `python app.py`)
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .scene import Scene


def _print_info(scene: Scene) -> None:
    s = scene.scale
    print(f"{len(scene)} objects | scale error (1 sigma): {100 * s.sigma_rel:.1f}%"
          f" | calibrated: {s.calibrated} | walls: {len(scene.layout.walls)}"
          f" | floor found: {scene.layout.floor_found}")
    print(f"{'id':<22}{'L x W x H (m)':<24}{'on':<16}{'views':<7}{'seen':<7}flags")
    for r in scene.summary():
        L, W, H = r["size_m"]
        print(f"{r['id']:<22}{f'{L:.2f} x {W:.2f} x {H:.2f}':<24}{str(r['support']):<16}"
              f"{r['views']:<7}{100 * r['observed_fraction']:>3.0f}%   {', '.join(r['flags'])}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scenelift", description="Film it. Edit it in 3D.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="reconstruct a scene from a video or photos")
    p.add_argument("input")
    p.add_argument("-o", "--out", required=True, help="scene folder to write")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--prompts", default=None, help="comma-separated object names to look for")
    p.add_argument("--device", default=None)
    p.add_argument("--sam3d-repo", default=None, help="path to a sam-3d-objects checkout (enables completion)")
    p.add_argument("--export", default=None, help="also export, e.g. scene.glb")
    p.add_argument("--reference", nargs=3, metavar=("OBJECT", "AXIS", "METRES"), default=None)

    for name, text in (("info", "list the objects of a scene"),
                       ("size", "length, width and height of an object, with error bars"),
                       ("place", "free floor spots where an object could stand")):
        q = sub.add_parser(name, help=text)
        q.add_argument("scene")
        if name != "info":
            q.add_argument("object")

    p = sub.add_parser("measure", help="distance between two objects")
    p.add_argument("scene"); p.add_argument("a"); p.add_argument("b")
    p.add_argument("--mode", choices=["surface", "center"], default="surface")

    p = sub.add_parser("fit", help="will an object fit?")
    p.add_argument("scene"); p.add_argument("object")
    p.add_argument("--space", nargs=3, type=float, metavar=("W", "D", "H"))
    p.add_argument("--between", nargs=2, metavar=("A", "B"))
    p.add_argument("--clearance", type=float, default=0.0)

    p = sub.add_parser("edit", help="remove, move, rotate or duplicate an object")
    p.add_argument("scene"); p.add_argument("op", choices=["remove", "move", "rotate", "duplicate"])
    p.add_argument("object"); p.add_argument("values", nargs="*", type=float)

    p = sub.add_parser("calibrate", help="fix scale from one known size")
    p.add_argument("scene"); p.add_argument("object")
    p.add_argument("axis", choices=["length", "width", "height"]); p.add_argument("metres", type=float)

    p = sub.add_parser("export", help="write the scene as .glb, .obj or .ply")
    p.add_argument("scene"); p.add_argument("path")
    p.add_argument("--background", action="store_true")

    p = sub.add_parser("eval", help="accuracy and calibration against tape-measure ground truth")
    p.add_argument("scene"); p.add_argument("ground_truth")

    p = sub.add_parser("studio", help="open SceneLift Studio, the browser UI")
    p.add_argument("--host", default=None, help="interface to bind (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="port (default 8000)")
    p.add_argument("--scenes-dir", default=None, help="folder that holds the scenes (default ./scenes)")
    p.add_argument("--device", default=None, help="torch device for the model backends")
    p.add_argument("--sam3d-repo", default=None, help="path to a sam-3d-objects checkout (enables completion)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")

    a = ap.parse_args(argv)

    if a.cmd == "studio":
        try:
            from .studio.server import serve, settings_from_args
        except ImportError as e:
            print(f"SceneLift Studio needs the web dependencies ({e}).\n"
                  f"Install them with:  pip install -r requirements.txt", file=sys.stderr)
            return 1
        serve(settings_from_args(a))
        return 0

    from . import query

    if a.cmd == "run":
        from .pipeline import SceneLift
        cfg = Config()
        if a.max_frames:
            cfg.max_frames = a.max_frames
        if a.prompts:
            cfg.prompts = [s.strip() for s in a.prompts.split(",") if s.strip()]
        lift = SceneLift.default(cfg, device=a.device, sam3d_repo=a.sam3d_repo,
                                 progress=lambda n, t: print(f"  {n:<10} {t:6.1f} s", file=sys.stderr))
        ref = (a.reference[0], a.reference[1], float(a.reference[2])) if a.reference else None
        scene = lift.run(a.input, reference=ref)
        scene.save(a.out)
        if a.export:
            scene.export(a.export)
        _print_info(scene)
        return 0

    try:
        scene = Scene.load(a.scene)
        if a.cmd == "info":
            _print_info(scene)
        elif a.cmd == "size":
            for m in query.dimensions(scene, a.object).values():
                print(m)
        elif a.cmd == "measure":
            print(query.distance(scene, a.a, a.b, a.mode))
        elif a.cmd == "fit":
            if a.between:
                print(query.fits_between(scene, a.object, a.between[0], a.between[1], a.clearance))
            elif a.space:
                print(query.fits_in_space(scene, a.object, a.space, clearance=a.clearance))
            else:
                ap.error("fit needs --space W D H or --between A B")
        elif a.cmd == "place":
            res = query.find_placements(scene, a.object)
            if not res:
                print("no free observed floor area is large enough")
            for r in res:
                print(f"x = {r['center_xy'][0]:.2f} m, y = {r['center_xy'][1]:.2f} m, "
                      f"turned {r['yaw_deg']:.0f} deg, spare {100 * r['spare_m']:.0f} cm")
        elif a.cmd == "edit":
            if a.op == "remove":
                scene.remove(a.object)
            elif a.op == "move":
                if len(a.values) != 3:
                    ap.error("move needs dx dy dz")
                scene.translate(a.object, a.values)
            elif a.op == "rotate":
                if len(a.values) != 1:
                    ap.error("rotate needs one angle in degrees")
                scene.rotate(a.object, a.values[0])
            elif a.op == "duplicate":
                print(scene.duplicate(a.object, tuple(a.values) if len(a.values) == 3 else (0.5, 0.0, 0.0)))
            scene.save(a.scene)
            _print_info(scene)
        elif a.cmd == "calibrate":
            f = scene.calibrate(query.dimensions(scene, a.object)[a.axis].value, a.metres)
            scene.save(a.scene)
            print(f"scale factor applied: {f:.4f}")
        elif a.cmd == "export":
            scene.export(a.path, include_background=a.background)
        elif a.cmd == "eval":
            from .eval import evaluate
            res = evaluate(scene, a.ground_truth)
            res.pop("rows", None)
            print(json.dumps(res, indent=2))
    except (KeyError, ValueError, OSError) as e:
        # a wrong object name, a bad value or a missing scene folder: say so in one line
        text = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"scenelift: error: {text}", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
