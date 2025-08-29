from __future__ import annotations
import sys, time, os
from dataclasses import dataclass

@dataclass
class Params:
    image_path: str
    dish_diameter: float
    dish_offset: float = 0.0
    cell_min_diameter: int = 0
    cell_max_diameter: int = sys.maxsize
    max_cells: int = 15
    nn_query_size: int = 10
    scale_for_hough: float = 0.25
    # morphology radii (in pixels)
    close_radius: int = 5
    open_radius: int = 1

@dataclass
class Weights:
    area: float = 2.0
    compactness: float = 1.0
    eccentricity: float = 0.25
    nn_collision_distance: float = 2.0
    solidity: float = 1.0

@dataclass
class Thresholds:
    min_compactness: float = 0.6
    min_solidity: float = 0.85
    max_eccentricity: float = 0.8
    std_weight_diameter: int = 3  # 0 disables

@dataclass
class OutputOpts:
    save_circle_detection: bool = True
    save_segmentation_process: bool = True
    save_cell_annotation: bool = True
    save_properties: bool = True
    plot_interactive_properties: bool = False  # default False to keep cv-free fast path
    outdir: str = "result"

def ensure_outdir(base: str) -> str:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(os.getcwd(), base, ts)
    os.makedirs(out, exist_ok=True)
    print(f"Saving artifacts to: {out}")
    return out
