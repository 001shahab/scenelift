# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Spatial answers for a language model, computed and never guessed.

A language model is good at turning "is there room for the box next to the
cabinet?" into a function call, and bad at producing the number. scenelift
gives it the functions (query.TOOLS) and runs them on the reconstruction
(query.call_tool). The number that reaches the user comes with an error bar.

This script shows the loop without calling any model, so it runs anywhere:

    python examples/llm_tools.py                # on the synthetic room
    python examples/llm_tools.py my_scene       # on a saved scene folder

To wire it to a real model, pass TOOLS as the tool definitions of your
provider's chat API, and feed each tool call the model makes to `answer()`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout without installing

from scenelift import Config, Scene, SceneLift, query  # noqa: E402


def load_scene(path=None) -> Scene:
    if path:
        return Scene.load(path)
    from scenelift.backends.synthetic import (SyntheticCompleter, SyntheticGeometry, SyntheticSegmenter,
                                              SyntheticWorld)

    world = SyntheticWorld(seed=0)
    lift = SceneLift(SyntheticGeometry(world), SyntheticSegmenter(world), SyntheticCompleter(world), Config())
    return lift.run_frames(world.frames())


def answer(scene: Scene, name: str, arguments: dict) -> str:
    """What you send back to the model as the result of its tool call."""
    try:
        return json.dumps(query.call_tool(scene, name, arguments))
    except KeyError as e:
        # an unknown or ambiguous object name: the message lists the candidates,
        # so the model can correct itself and call again
        return json.dumps({"error": str(e.args[0])})


def main() -> None:
    scene = load_scene(sys.argv[1] if len(sys.argv) > 1 else None)

    print("Tool definitions to give to the model:\n")
    print(json.dumps(query.TOOLS, indent=2))

    # The calls a model would make for three questions about this scene.
    ids = scene.ids
    calls = [("list_objects", {})]
    if ids:
        calls.append(("get_dimensions", {"object": ids[0]}))
        calls.append(("check_fit_in_space", {"object": ids[0], "width": 1.0, "depth": 0.6, "height": 1.2}))
    if len(ids) >= 2:
        calls.append(("measure_distance", {"a": ids[0], "b": ids[1]}))
    calls.append(("get_dimensions", {"object": "unicorn"}))  # a name that is not in the scene

    for name, arguments in calls:
        print(f"\nmodel calls  {name}({json.dumps(arguments)})")
        print(f"tool returns {answer(scene, name, arguments)}")


if __name__ == "__main__":
    main()
