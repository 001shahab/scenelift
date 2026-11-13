# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Quick start: reconstruct, measure, ask, edit, export.

Without arguments this runs on the synthetic room, on a laptop CPU, with no
model weights:

    python examples/quickstart.py

With a video or a folder of photos it uses the real model backends, which must
be installed first (docs/BACKENDS.md):

    python examples/quickstart.py room.mp4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout without installing

from scenelift import Config, SceneLift, query, select  # noqa: E402


def reconstruct(source=None):
    if source is not None:
        return SceneLift.default().run(source)
    # the synthetic room: exact depth, then a scale error per view and pixel noise
    from scenelift.backends.synthetic import (SyntheticCompleter, SyntheticGeometry, SyntheticSegmenter,
                                              SyntheticWorld)

    world = SyntheticWorld(seed=0)
    lift = SceneLift(SyntheticGeometry(world), SyntheticSegmenter(world), SyntheticCompleter(world), Config())
    return lift.run_frames(world.frames())


def main() -> None:
    scene = reconstruct(sys.argv[1] if len(sys.argv) > 1 else None)

    print("\nObjects")
    for row in scene.summary():
        size = " x ".join(f"{s:.2f}" for s in row["size_m"])
        print(f"  {row['id']:<12} {size} m, rests on {row['support']}, "
              f"{row['observed_fraction']:.0%} observed")
    print(f"\nScale: any length in this scene is known to +/- {scene.scale.sigma_rel:.1%} (1 sigma)")

    ids = scene.ids
    if not ids:
        print("No objects were found.")
        return
    first = ids[0]

    print("\nSizes, each with an error bar")
    for m in query.dimensions(scene, first).values():
        print(f"  {m}")

    if len(ids) >= 2:
        print("\nDistance")
        print(f"  {query.distance(scene, ids[0], ids[1])}")

    print("\nWill it fit in a space of 1.0 x 0.6 x 1.2 m?")
    print(f"  {first}: {query.fits_in_space(scene, first, (1.0, 0.6, 1.2))}")

    print("\nWhere could it stand?")
    for p in query.find_placements(scene, first)[:3]:
        print(f"  x {p['center_xy'][0]:+.2f}, y {p['center_xy'][1]:+.2f} m, spare {p['spare_m']:.2f} m")

    print("\nWhat is in the middle of the first photo?")
    height, width = scene.cameras[0]["size"]
    print(f"  {select.by_click(scene, 0, width / 2, height / 2)}")

    print("\nEdit: move the first object 30 cm along X. Whatever rests on it moves too.")
    scene.translate(first, (0.3, 0.0, 0.0))

    # ./scenes is the folder SceneLift Studio reads by default
    out = Path(__file__).resolve().parents[1] / "scenes" / "quickstart"
    scene.save(out)
    scene.export(out / "scene.glb")
    print(f"\nSaved to {out} and exported scene.glb next to it.")
    print("Open it in SceneLift Studio:  python app.py")


if __name__ == "__main__":
    main()
