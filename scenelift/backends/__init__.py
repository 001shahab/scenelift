# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Model backends. The core of scenelift never imports a model directly; it
talks to these three small interfaces, so any backbone can be swapped."""

from .base import CompletionBackend, GeometryBackend, SegmentationBackend

__all__ = ["GeometryBackend", "SegmentationBackend", "CompletionBackend"]
