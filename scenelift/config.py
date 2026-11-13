# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""All tunable numbers in one place. Every default is in metres or pixels."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


DEFAULT_PROMPTS = [
    "chair", "sofa", "armchair", "table", "desk", "bed", "nightstand", "cabinet",
    "shelf", "wardrobe", "tv", "monitor", "laptop", "lamp", "plant", "vase",
    "mug", "bottle", "book", "box", "bag", "pillow", "rug", "stool", "bench",
    "refrigerator", "oven", "microwave", "sink", "toilet", "bathtub", "mirror",
    "picture frame", "trash can", "speaker", "keyboard", "toy",
]


@dataclass
class Config:
    # ---- ingest ----
    max_frames: int = 24  # keyframes kept from a video
    frame_stride: int = 3  # look at every n-th video frame
    min_flow_px: float = 12.0  # min median optical flow between keyframes (at 320 px width)
    max_image_side: int = 1280  # longer side of stored frames

    # ---- geometry filtering ----
    min_conf: float = 0.3  # drop pixels below this confidence
    depth_edge_rel: float = 0.04  # drop pixels whose depth jumps more than 4% to a neighbour
    max_depth: float = 12.0  # ignore anything farther than this

    # ---- cross-view scale harmonisation ----
    scale_points_per_view: int = 20000
    scale_gate: float = 0.25  # |log depth ratio| gate that rejects occlusions
    scale_min_matches: int = 300
    backbone_scale_sigma: float = 0.04  # prior relative error of the backbone's absolute scale

    # ---- gravity / layout ----
    plane_thresh: float = 0.02
    plane_iters: int = 400
    floor_max_tilt_deg: float = 35.0
    wall_min_fraction: float = 0.04
    max_walls: int = 4

    # ---- segmentation / association ----
    prompts: list = field(default_factory=lambda: list(DEFAULT_PROMPTS))
    mask_min_pixels: int = 150
    mask_erode_px: int = 1
    mask_nms_iou: float = 0.7
    assoc_voxel: float = 0.05
    assoc_overlap: float = 0.25
    assoc_projection: float = 0.7  # share of one track's points that must fall in the other's masks
    min_views_per_object: int = 2  # automatically relaxed to 1 for 1-2 input views
    min_object_points: int = 80

    # ---- fusion ----
    fuse_voxel: float = 0.01
    background_voxel: float = 0.025
    sor_k: int = 12
    sor_std: float = 2.0

    # ---- completion ----
    icp_points: int = 2500
    icp_iters: int = 30
    icp_trim: float = 0.85
    observed_tau: float = 0.025  # a mesh vertex within this distance of a measured point is "observed"
    freespace_tol: float = 0.04
    max_freespace_violation: float = 0.25  # reject a completion that contradicts this much free space
    silhouette_scales: tuple = (0.94, 1.20, 14)  # (min, max, steps) of the scale search
    silhouette_views: int = 6
    # accept a generated shape only if (outline IoU - free-space violation) reaches this.
    # Tuned on synthetic shapes; recalibrate on the real benchmark.
    min_fit_quality: float = 0.72
    max_fit_residual: float = 0.03
    max_fit_residual_rel: float = 0.05

    # ---- physics ----
    support_tol: float = 0.06
    max_push: float = 0.04

    # ---- uncertainty model ----
    sigma_local: float = 0.008  # local surface noise of fused geometry
    generated_sigma_frac: float = 0.15  # sigma of an unseen extent = this * extent (calibrate on the benchmark)
    fit_yes: float = 0.95
    fit_no: float = 0.05

    # ---- misc ----
    seed: int = 0
    save_depth: bool = True
    save_frames: bool = True  # keep the keyframes with the scene (frames/<index>.jpg), for viewers

    def to_dict(self) -> dict:
        return asdict(self)
