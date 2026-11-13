# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""The three interfaces a backend can implement."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..types import CompletedShape, Frame, Mask2D, View


@runtime_checkable
class GeometryBackend(Protocol):
    """Images in, metric depth + intrinsics + camera poses out."""

    def infer(self, frames: list[Frame]) -> list[View]:
        ...


@runtime_checkable
class SegmentationBackend(Protocol):
    """For every view, a list of instance masks at the view's resolution.
    Set Mask2D.track_id when the model follows objects across frames."""

    def segment(self, views: list[View]) -> list[list[Mask2D]]:
        ...


@runtime_checkable
class CompletionBackend(Protocol):
    """One image + one mask in, a complete mesh in any frame and scale out.
    Return None when the model cannot handle the object."""

    def complete(self, view: View, mask: Mask2D) -> Optional[CompletedShape]:
        ...
