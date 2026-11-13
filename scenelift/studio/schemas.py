# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Request bodies of the Studio REST API. Lengths are metres, angles degrees,
pixels (u, v) = (column, row) of the camera's own image size."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

Vec3 = tuple[float, float, float]
Axis = Literal["length", "width", "height"]


class DemoRequest(BaseModel):
    """Build the synthetic room of scenelift.backends.synthetic. No GPU, no weights."""

    name: str = Field("Synthetic room", max_length=80)
    n_views: int = Field(10, ge=2, le=24, description="cameras on a 130 degree arc")
    scale_jitter: float = Field(0.05, ge=0.0, le=0.2, description="spread of the per-view scale error")
    depth_noise: float = Field(0.003, ge=0.0, le=0.03, description="relative pixel noise on depth")
    seed: int = Field(0, ge=0, le=2**31 - 1)
    completer: Literal["correct", "wrong", "none"] = Field(
        "correct",
        description="correct: the right shape in a wrong pose and scale. wrong: a sphere for every "
                    "object, which must be rejected. none: no generative backend, convex hulls only.")


class EditRequest(BaseModel):
    """One edit of the scene. `drop` lowers an object onto whatever is beneath
    it; `settle` re-runs the support and gravity reasoning of step 9."""

    op: Literal["remove", "translate", "rotate", "duplicate", "drop", "settle"]
    object: Optional[str] = Field(None, description="object id; not needed for settle")
    delta: Optional[Vec3] = Field(None, description="translate: (dx, dy, dz)")
    yaw_deg: Optional[float] = Field(None, description="rotate: angle about the vertical axis")
    offset: Vec3 = Field((0.5, 0.0, 0.0), description="duplicate: where the copy goes, relative to the original")
    carry: bool = Field(True, description="translate, rotate, drop: also move whatever rests on the object")
    settle: Optional[bool] = Field(
        None, description="re-run physics after the edit. Default: as the library does, "
                          "which is yes for remove and no for the other edits.")


class CalibrateRequest(BaseModel):
    """Fix the absolute scale from one known length. Give either an object and
    an axis, or the length as the scene currently reports it."""

    true_m: float = Field(..., gt=0.0)
    object: Optional[str] = None
    axis: Axis = "height"
    measured_m: Optional[float] = Field(None, gt=0.0)


class ObjectRequest(BaseModel):
    object: str


class DistanceRequest(BaseModel):
    a: str
    b: str
    mode: Literal["surface", "center"] = "surface"


class FitSpaceRequest(BaseModel):
    object: str
    space: Vec3 = Field(..., description="(width, depth, height) of the space")
    allow_rotation: bool = True
    clearance: float = Field(0.0, ge=0.0)


class FitBetweenRequest(BaseModel):
    object: str
    a: str
    b: str
    clearance: float = Field(0.0, ge=0.0)


class PlacementsRequest(BaseModel):
    object: Optional[str] = None
    footprint: Optional[tuple[float, float]] = Field(None, description="(length, width) when no object is given")
    height: Optional[float] = Field(None, gt=0.0)
    clearance: float = Field(0.02, ge=0.0)
    max_results: int = Field(5, ge=1, le=20)


class ClickRequest(BaseModel):
    camera: int = Field(..., ge=0)
    u: float
    v: float


class BoxRequest(BaseModel):
    camera: int = Field(..., ge=0)
    box: tuple[float, float, float, float] = Field(..., description="(u0, v0, u1, v1)")


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200)


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)


class EvaluateRequest(BaseModel):
    ground_truth: list[dict] = Field(..., description="the JSON list read by scenelift.eval.evaluate")
