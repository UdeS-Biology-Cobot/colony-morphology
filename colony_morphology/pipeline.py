from __future__ import annotations
import pandas as pd
from imageio.v3 import imread
from .config import Params, Weights, Thresholds, OutputOpts, ensure_outdir
from .dish import detect_dish_circle
from .mask import mask_and_crop
from .segment import preprocess_and_segment
from .props import compute_region_properties, compute_nn_metrics
from .score import score_and_filter, select_top_indices
from .outputs import (
    annotate_best_cells, save_segmentation_mosaic, save_scalar_images,
    draw_circle_debugs, plot_region_properties_interactive
)
from skimage.io import imsave

def run_from_array(image_rgb, params: Params, weights: Weights, thr: Thresholds, out: OutputOpts | None = None):
    """
    Pure array pipeline. Returns (top_idx, props, (x_min,y_min), outdir_or_None)
    """
    assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3, "image_rgb must be HxWx3 RGB"

    outdir = ensure_outdir(out.outdir) if out else None

    # 1) dish on grayscale u8
    from skimage.color import rgb2gray
    from skimage.util import img_as_ubyte
    img_gray_u8_full = img_as_ubyte(rgb2gray(image_rgb))
    cx, cy, r = detect_dish_circle(img_gray_u8_full, params.dish_diameter, params.scale_for_hough)

    if out and out.save_circle_detection:
        r_s = int(r * params.scale_for_hough)
        cy_s = int(cy * params.scale_for_hough)
        cx_s = int(cx * params.scale_for_hough)
        draw_circle_debugs(image_rgb, params.scale_for_hough, (cy_s, cx_s, r_s), (cx, cy, r), adjusted_points=None, outdir=outdir)

    # 2) mask + crop
    img_crop_rgb, img_orig_crop_rgb, mask_thresh_crop, (x_min, y_min) = mask_and_crop(image_rgb, (cx, cy, r), params.dish_offset)

    # 3) preprocess + segment
    assets = preprocess_and_segment(img_crop_rgb,
                                    mask_thresh_crop,
                                    close_radius=max(0, int(params.close_radius)),
                                    open_radius=max(0, int(params.open_radius)),)
    labels = assets["labels"]

    # 4) props + nn + scoring
    props = compute_region_properties(assets["img_gray"], labels)
    compute_nn_metrics(props, params.nn_query_size)
    quality_metrics, props = score_and_filter(props, weights, thr, params)
    top_idx = select_top_indices(quality_metrics, params.max_cells)

    # 5) outputs
    if out:
        if out.save_cell_annotation and top_idx:
            annotate_best_cells(img_orig_crop_rgb, props, top_idx, f"{outdir}/annotated_cell.png")
        if out.save_segmentation_process:
            save_segmentation_mosaic(assets, img_orig_crop_rgb, labels, f"{outdir}/segmentation.png")
            imsave(f'{outdir}/crop.png', img_orig_crop_rgb)
            save_scalar_images(assets, outdir)
        if out.save_properties:
            from colony_morphology import regionprops_util as cb
            import numpy as np
            prop_names = (
                'label','cell_quality','compactness','nn_centroid_distance','nn_collision_distance',
                'discarded','discarded_description','axes_closness','area','area_bbox','area_convex',
                'area_filled','axis_major_length','axis_minor_length','bbox','centroid','centroid_local',
                'centroid_weighted','centroid_weighted_local','coords','coords_scaled','eccentricity',
                'equivalent_diameter_area','euler_number','extent','feret_diameter_max','image','image_convex',
                'image_filled','image_intensity','inertia_tensor','inertia_tensor_eigvals','intensity_max',
                'intensity_mean','intensity_min','label','moments','moments_central','moments_hu',
                'moments_normalized','moments_weighted','moments_weighted_central','moments_weighted_hu',
                'moments_weighted_normalized','num_pixels','orientation','perimeter','perimeter_crofton',
                'slice','solidity'
            )
            d = cb.regionprops_to_dict(props, prop_names)
            max_len = max((len(getattr(p, "discarded_description", "")) for p in props), default=0)
            d["discarded_description"] = np.empty(len(props), dtype=f'<U{max_len}')
            for i, p in enumerate(props):
                d["discarded_description"][i] = getattr(p, "discarded_description", "")
            pd.DataFrame(d).to_excel(f'{outdir}/region_properties.xlsx', index=False)

        if out.plot_interactive_properties:
            property_names = [
                'area','eccentricity','perimeter','solidity','compactness','axes_closness',
                'equivalent_diameter_area','nn_collision_distance','cell_quality','discarded','discarded_description'
            ]
            plot_region_properties_interactive(img_orig_crop_rgb, labels, props, property_names)

    return top_idx, props, (x_min, y_min), outdir





def run(params: Params, weights: Weights, thr: Thresholds, out: OutputOpts | None, *, return_offsets: bool = False):
    """
    File-based wrapper that reuses run_from_array.
    Returns:
      default: (top_idx, props, outdir)
      if return_offsets=True: (top_idx, props, (x_min,y_min), outdir)
    """
    img = imread(params.image_path, exifrotate=True)
    assert img is not None, f"Cannot read file: {params.image_path}"
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]  # drop alpha
    top_idx, props, offsets, outdir = run_from_array(img, params, weights, thr, out)
    return (top_idx, props, offsets, outdir) if return_offsets else (top_idx, props, outdir)
