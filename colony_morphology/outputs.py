from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import circle_perimeter
from skimage.transform import rescale
from skimage.util import img_as_ubyte
from skimage.io import imsave
from skimage import measure
import plotly.graph_objects as go
import plotly.io as pio

def draw_circle_debugs(img_rgb, scale, coarse_circle, refined_circle, adjusted_points, outdir):
    cy_f, cx_f, r_f = coarse_circle
    cy_a, cx_a, r_a = refined_circle[1], refined_circle[0], refined_circle[2]

    # scaled
    simg = np.clip(rescale(img_rgb, scale=scale, anti_aliasing=True, channel_axis=-1), 0, 1).copy()
    rr, cc = circle_perimeter(int(cy_f), int(cx_f), int(r_f), shape=simg.shape)
    simg[rr, cc] = (0.0, 1.0, 0.2)
    imsave(f"{outdir}/circle_detection_scaled.png", img_as_ubyte(simg))

    # original
    oimg = img_rgb.copy()
    rr2, cc2 = circle_perimeter(int(cy_a), int(cx_a), int(r_a), shape=oimg.shape)
    oimg[rr2, cc2] = (0, 255, 51)
    if adjusted_points is not None:
        for pt in adjusted_points:
            oimg[int(pt[1]), int(pt[0])] = (255, 0, 0)
    imsave(f"{outdir}/circle_detection_original.png", oimg)

def annotate_best_cells(img_rgb, properties, top_indices, outpath):
    fig, ax = plt.subplots()
    ax.imshow(img_rgb); ax.axis("off")
    for rank, idx in enumerate(top_indices, start=1):
        p = properties[idx]
        r = p["equivalent_diameter_area"]/2.0 + 5.0
        y, x = p.centroid
        circ = plt.Circle((x, y), radius=r, fc='none', color='red')
        ax.add_patch(circ)
        ax.annotate(rank, xy=(x + r, y - r), color='red')
    h, w = img_rgb.shape[:2]; dpi = 300
    fig.set_size_inches(w / dpi, h / dpi)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_alpha(0)
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0, dpi=dpi, transparent=True)
    plt.close(fig)

def save_segmentation_mosaic(assets, img_orig_crop_rgb, labels, outpath_png):
    layout = [["A","B","C"],
              ["D","E","F"],
              ["G","H","I"]]
    fig, axd = plt.subplot_mosaic(layout, constrained_layout=True, dpi=300)
    axd['A'].imshow(img_orig_crop_rgb); axd['A'].set_title('Crop'); axd['A'].set_axis_off()
    axd['B'].imshow(assets["img_gray"], cmap=plt.cm.gray); axd['B'].set_title('Mask'); axd['B'].set_axis_off()
    axd['C'].imshow(assets["img_mask_artifacts"], cmap=plt.cm.gray); axd['C'].set_title('Threshold'); axd['C'].set_axis_off()
    axd['D'].imshow(assets["img_mask_artifacts"], cmap=plt.cm.gray); axd['D'].set_title('Filter'); axd['D'].set_axis_off()
    axd['E'].imshow(assets["closing"], cmap=plt.cm.gray); axd['E'].set_title('Closing'); axd['E'].set_axis_off()
    axd['F'].imshow(assets["opening"], cmap=plt.cm.gray); axd['F'].set_title('Opening'); axd['F'].set_axis_off()
    axd['G'].imshow(assets["dist"], cmap=plt.cm.gray); axd['G'].set_title('Distance'); axd['G'].set_axis_off()
    axd['H'].imshow(labels, cmap=plt.cm.nipy_spectral); axd['H'].set_title('Watershed'); axd['H'].set_axis_off()
    plt.savefig(outpath_png); plt.close(fig)

def save_scalar_images(assets, outdir):
    imsave(f'{outdir}/mask.png', assets["img_gray"])
    imsave(f'{outdir}/threshold.png', img_as_ubyte(assets["img_mask_artifacts"]))
    imsave(f'{outdir}/closing.png', img_as_ubyte(assets["closing"]))
    imsave(f'{outdir}/opening.png', img_as_ubyte(assets["opening"]))
    dist = assets["dist"]
    dist_norm = dist / np.max(dist) if np.max(dist) > 0 else dist
    imsave(f'{outdir}/distance.png', img_as_ubyte(dist_norm))
    plt.imsave(f'{outdir}/watershed.png', assets["labels"], cmap='nipy_spectral')

def plot_region_properties_interactive(img_rgb, labels, properties, property_names):
    """
    cv-free reimplementation of your interactive overlay.
    """
    fig = go.Figure()
    h, w = img_rgb.shape[:2]
    fig.add_trace(go.Image(z=img_rgb))
    for p in properties:
        contour = measure.find_contours(labels == p.label, 0.5)[0]
        y, x = contour.T
        hover = []
        for name in property_names:
            val = getattr(p, name)
            if isinstance(val, bool):
                hover.append(f"<b>{name}:</b> {val}")
            elif isinstance(val, str):
                val = val.replace("\n", "<br>&nbsp;&nbsp;")
                hover.append(f"<b>{name}:</b> {val}")
            else:
                try:
                    hover.append(f"<b>{name}:</b> {val:.3f}")
                except Exception:
                    hover.append(f"<b>{name}:</b> {val}")
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=str(p.label),
            fill='toself', showlegend=False,
            hovertemplate='<br>'.join(hover), hoveron='points+fills'
        ))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
    pio.show(fig)
