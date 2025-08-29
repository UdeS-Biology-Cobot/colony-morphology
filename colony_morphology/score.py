from __future__ import annotations
import statistics, numpy as np
from .config import Weights, Thresholds, Params

def score_and_filter(properties, weights: Weights, thresholds: Thresholds, params: Params):
    # normalize stats from reasonable cells
    good_nn = [p["nn_collision_distance"] for p in properties if p["compactness"] >= 0.2 and p["nn_collision_distance"] >= 0]
    good_area = [p["area"] for p in properties if p["compactness"] >= 0.2]
    max_nn = max(good_nn) if good_nn else 1.0
    max_area = max(good_area) if good_area else 1.0

    quality_metrics = np.empty(len(properties), dtype=object)
    used = sum(v != 0 for v in (weights.area, weights.compactness, weights.eccentricity, weights.nn_collision_distance, weights.solidity))
    if not used:
        raise ValueError("All metric weights are zero.")

    for i, p in enumerate(properties):
        n_area = p.area / max_area
        n_nn = (p.nn_collision_distance / max_nn) if max_nn else 0.0
        n_comp = min(p.compactness, 1.0)
        n_ecc = 1.0 - p.eccentricity

        score = (weights.area * n_area +
                 weights.compactness * n_comp +
                 weights.eccentricity * n_ecc +
                 weights.nn_collision_distance * n_nn +
                 weights.solidity * p.solidity) / used

        # Threshold-based discards (use equivalent_diameter_area consistently)
        def discard(msg):
            p.discarded = True
            p.discarded_description += msg + "\n"
            return 0.0

        if params.cell_min_diameter and p.equivalent_diameter_area < params.cell_min_diameter:
            score = discard(f"Cell equivalent diameter {p.equivalent_diameter_area:.2f} < {params.cell_min_diameter:.2f}")
        if params.cell_max_diameter and p.equivalent_diameter_area > params.cell_max_diameter:
            score = discard(f"Cell equivalent diameter {p.equivalent_diameter_area:.2f} > {params.cell_max_diameter:.2f}")
        if thresholds.min_compactness and p.compactness < thresholds.min_compactness:
            score = discard(f"Compactness {p.compactness:.2f} < {thresholds.min_compactness:.2f}")
        if thresholds.min_solidity and p.solidity < thresholds.min_solidity:
            score = discard(f"Solidity {p.solidity:.2f} < {thresholds.min_solidity:.2f}")
        if thresholds.max_eccentricity and p.eccentricity > thresholds.max_eccentricity:
            score = discard(f"Eccentricity {p.eccentricity:.2f} > {thresholds.max_eccentricity:.2f}")
        if getattr(p, "nn_collision_distance", 0) < 0:
            score = discard(f"In collision. Distance: {p.nn_collision_distance:.2f}")

        p.cell_quality = score
        quality_metrics[i] = (score, i)

    # std outlier removal on equivalent_diameter_area
    if thresholds.std_weight_diameter:
        kept = [p["equivalent_diameter_area"] for p in properties if p.cell_quality > 0.0]
        if kept:
            d_std = statistics.pstdev(kept)
            d_mean = statistics.mean(kept)
            lo = d_mean - thresholds.std_weight_diameter * d_std
            hi = d_mean + thresholds.std_weight_diameter * d_std
            for i, p in enumerate(properties):
                if p.equivalent_diameter_area < lo or p.equivalent_diameter_area > hi:
                    p.discarded = True
                    p.discarded_description += (
                        f"Diameter {p.equivalent_diameter_area:.2f} outside [{lo:.2f}, {hi:.2f}], "
                        f"mean={d_mean:.2f}, sigma={thresholds.std_weight_diameter:.2f}, std={d_std:.2f}\n"
                    )
                    p.cell_quality = 0.0
                    quality_metrics[i] = (0.0, i)

    return quality_metrics, properties

def select_top_indices(quality_metrics, max_cells: int):
    ranked = sorted(quality_metrics, key=lambda x: x[0], reverse=True)
    if max_cells and len(ranked) > max_cells:
        ranked = ranked[:max_cells]
    return [idx for _, idx in ranked]
