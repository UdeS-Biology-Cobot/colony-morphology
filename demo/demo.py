from __future__ import annotations
import argparse

from colony_morphology.config import Params, Weights, Thresholds, OutputOpts
from colony_morphology.pipeline import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Demo runner for colony_morphologie")
    # Inputs
    p.add_argument("--image", "--image_path", dest="image_path", required=True, type=str,
                   help="Path to the input RGB image (PNG/JPG/TIF).")
    p.add_argument("--dish-diameter", dest="dish_diameter", required=True, type=float,
                   help="Known Petri dish diameter in pixels (outer rim).")
    p.add_argument("--dish-offset", dest="dish_offset", default=0.0, type=float,
                   help="Offset to shrink the usable dish radius (pixels).")

    # Size constraints / neighbor query
    p.add_argument("--cell-min-diameter", dest="cell_min_diameter", default=0, type=int,
                   help="Discard cells smaller than this equivalent diameter (0 to disable).")
    p.add_argument("--cell-max-diameter", dest="cell_max_diameter", default=2**31-1, type=int,
                   help="Discard cells larger than this equivalent diameter (0 to disable).")
    p.add_argument("--max-cells", dest="max_cells", default=15, type=int,
                   help="How many top cells to keep in the final ranking.")
    p.add_argument("--nn-query-size", dest="nn_query_size", default=10, type=int,
                   help="k for nearest-neighbor query when estimating collision distance.")

    # Erosion and dilatation, set to 0 to skip
    p.add_argument("--close-radius", type=int, default=5)
    p.add_argument("--open-radius", type=int, default=1)


    # Weights
    p.add_argument("--w-area", dest="w_area", default=2.0, type=float)
    p.add_argument("--w-compactness", dest="w_compactness", default=1.0, type=float)
    p.add_argument("--w-eccentricity", dest="w_eccentricity", default=0.25, type=float)
    p.add_argument("--w-nn-collision", dest="w_nn_collision", default=2.0, type=float)
    p.add_argument("--w-solidity", dest="w_solidity", default=1.0, type=float)

    # Thresholds / statistics
    p.add_argument("--min-compactness", dest="min_compactness", default=0.6, type=float)
    p.add_argument("--min-solidity", dest="min_solidity", default=0.85, type=float)
    p.add_argument("--max-eccentricity", dest="max_eccentricity", default=0.8, type=float)
    p.add_argument("--std-weight-diameter", dest="std_weight_diameter", default=3, type=int,
                   help="Std-deviation filter on equivalent diameter (0 to disable).")

    # Outputs
    p.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=True, help="Enable verbose logging with per-step timings.")
    p.add_argument("--outdir", dest="outdir", default="result", type=str,
                   help="Directory to create timestamped run folder in.")
    p.add_argument("--save-circle-detection", dest="save_circle_detection", action="store_true", default=True)
    p.add_argument("--no-save-circle-detection", dest="save_circle_detection", action="store_false")
    p.add_argument("--save-segmentation-process", dest="save_segmentation_process", action="store_true", default=True)
    p.add_argument("--no-save-segmentation-process", dest="save_segmentation_process", action="store_false")
    p.add_argument("--save-cell-annotation", dest="save_cell_annotation", action="store_true", default=True)
    p.add_argument("--no-save-cell-annotation", dest="save_cell_annotation", action="store_false")
    p.add_argument("--save-properties", dest="save_properties", action="store_true", default=True)
    p.add_argument("--no-save-properties", dest="save_properties", action="store_false")
    p.add_argument("--plot-interactive-properties", dest="plot_interactive_properties", action="store_true", default=False,
                   help="Open interactive Plotly overlay (may be slow).")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    params = Params(
        image_path=args.image_path,
        dish_diameter=args.dish_diameter,
        dish_offset=args.dish_offset,
        cell_min_diameter=args.cell_min_diameter,
        cell_max_diameter=args.cell_max_diameter,
        max_cells=args.max_cells,
        nn_query_size=args.nn_query_size,
        scale_for_hough=0.25,
        close_radius=args.close_radius,
        open_radius=args.open_radius,
    )
    weights = Weights(
        area=args.w_area,
        compactness=args.w_compactness,
        eccentricity=args.w_eccentricity,
        nn_collision_distance=args.w_nn_collision,
        solidity=args.w_solidity,
    )
    thresholds = Thresholds(
        min_compactness=args.min_compactness,
        min_solidity=args.min_solidity,
        max_eccentricity=args.max_eccentricity,
        std_weight_diameter=args.std_weight_diameter,
    )
    outputs = OutputOpts(
        save_circle_detection=args.save_circle_detection,
        save_segmentation_process=args.save_segmentation_process,
        save_cell_annotation=args.save_cell_annotation,
        save_properties=args.save_properties,
        plot_interactive_properties=args.plot_interactive_properties,
        outdir=args.outdir,
    )

    # Use the file-based wrapper which delegates to run_from_array under the hood.
    top_idx, props, outdir = run(params, weights, thresholds, outputs, args.verbose)

    print("")
    print("=== Top colonies ===")
    for rank, idx in enumerate(top_idx, start=1):
        p = props[idx]
        # note: regionprops attrs available via both key and attribute access
        area = p["area"]
        qual = getattr(p, "cell_quality", 0.0)
        diam = p["equivalent_diameter_area"]
        centroid = p["centroid"]
        print(f"{rank:2d}. label={p.label:4d} area={area:7.1f} diam={diam:7.2f} quality={qual:6.3f} centroid=(y={centroid[0]:.1f}, x={centroid[1]:.1f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
