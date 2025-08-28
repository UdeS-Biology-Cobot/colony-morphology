#!/usr/bin/env python3
from colony_morphology.geometry import *
from colony_morphology.image_transform import *
from colony_morphology.plotting import plot_bboxes, plot_region_roperties
from colony_morphology.skimage_util import compactness as compactness_cb
from colony_morphology.skimage_util import nn_centroid_distance as nn_centroid_distance_cb
from colony_morphology.skimage_util import nn_collision_distance as nn_collision_distance_cb
from colony_morphology.skimage_util import cell_quality as cell_quality_cb
from colony_morphology.skimage_util import axes_closness as axes_closness_cb
from colony_morphology.skimage_util import discarded as discarded_cb
from colony_morphology.skimage_util import discarded_description as discarded_description_cb
from colony_morphology.skimage_util import regionprops_to_dict
from skimage.io import imsave

from colony_morphology.metric import compactness as compute_compactness
from colony_morphology.metric import axes_closness as compute_axes_closness

from scipy.optimize import leastsq

import argparse
import cv2 as cv
from io import BytesIO
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import color, io, morphology
from skimage.feature import peak_local_max
from skimage.filters import try_all_threshold, threshold_yen, threshold_otsu, threshold_local
from skimage.measure import regionprops, regionprops_table, label
from skimage.segmentation import watershed
from skimage.transform import hough_circle, hough_circle_peaks, rescale
from skimage.draw import circle_perimeter, circle_perimeter_aa, line
from skimage.util import img_as_ubyte
from numpy.linalg import norm
import statistics
import sys
import time

if __name__=="__main__":

    # Parsing
    parser = argparse.ArgumentParser(
        description="Script that finds best colonies to pick"
    )

    parser.add_argument("--image_path", required=True, type=str)
    parser.add_argument("--dish_diameter", required=True, type=float)
    parser.add_argument("--dish_offset", required=False, type=float, default=0)
    parser.add_argument("--cell_min_diameter", required=False, type=int, default=0)
    parser.add_argument("--cell_max_diameter", required=False, type=int, default=sys.maxsize)
    parser.add_argument("--max_cells", required=False, type=int, default=15)
    args = parser.parse_args()

    # Parameters
    image_path = args.image_path
    dish_diameter = args.dish_diameter
    dish_offset =  args.dish_offset
    cell_min_diameter = args.cell_min_diameter
    cell_max_diameter = args.cell_max_diameter
    max_cells = args.max_cells

    weight_area = 2.0                   # Area of the region i.e. number of pixels of the region scaled by pixel-area
    weight_compactness = 1.0            # Ratio of the area of an object to the area of a circle with the same perimeter
    weight_eccentricity = 0.25          # Eccentricity of the ellipse that has the same second-moments as the region
    weight_nn_collision_distance = 2.0  # Nearest neighbor distance [pixel]
    weight_solidity = 1.0               # Ratio of pixels in the region to pixels of the convex hull image

    # nearest neighbor
    nn_query_size = 10  # Return X nearest neighbor's from the centroid tree query, min = 1
                        # A higher number will lead to a better aproximation of the collision distance


    # Discard cells based on Threshold, set to 0 to disable
    cell_min_compactness = 0.6      # discard if < min_compactness [pixel]
    cell_min_solidity = 0.85        # discard if < min_solidity [pixel]
    cell_max_eccentricity = 0.8     # discard if > max_eccentricity [pixel]

    # Discard cells based on Statistics, set to 0 to disable
    # The mean and std are computed on non discarded cell's (after passing through the Threshold filtering)
    std_weight_diameter = 3   # 1 = Keep 68.27% of cell's between mu -1*sigma and mu + 1*sigma
                                # 2 = Keep 95.45% of cell's between mu -2*sigma and mu + 2*sigma
                                # 3 = Keep 99.73% of cell's between mu -3*sigma and mu + 3*sigma

    # Morphological cleanup for high-res masks: when adaptive thresholding isn’t perfect,
    # an elliptical closing fills tiny gaps/holes and a small opening removes speckle noise.
    # (Ellipse matches round colonies; tune sizes to your imaging scale.)
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE,(1,1))
    kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE,(1,1))

    absolute_path = image_path
    if (not os.path.isabs(image_path)):
        absolute_path = os.path.join(os.getcwd(), image_path)


    save_circle_detection = True      # subplot of circle detection algorithm [png]
    save_segmentation_process = True  # subplot of all the segmentation (after circle detection) [png]
    save_cell_annotation = True       # annotated cell wrt. the cell_quality metric [png]
    save_properties = True           # metrics of region properties [xlsx]

    plot_interactive_properties = True  # plot cell properties on a web client, web page will open automatically


    output_path = os.path.join(os.getcwd(), "result/")
    output_path += time.strftime("%Y-%m-%d_%H-%M-%S") #avoid name clash

    if(save_circle_detection or save_segmentation_process or save_cell_annotation or save_properties):
        # Create the directory
        try:
            os.mkdir(output_path)
            print(f'Saving images to: {output_path}')
        except FileExistsError:
            print(f"Directory '{output_path}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{output_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")


    # Read image
    print(f'Reading image from: {absolute_path}')
    img_bgr = cv.imread(image_path)
    assert img_bgr is not None, "File could not be read, check with os.path.exists()"

    # Compute total algorithm processingtime, excluding plotting
    global_time = 0
    global_start = time.time()


    # 1- Convert to grayscale
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    # Mask image to contain only the petri dish
    # 1a- Rescale the image to reduce memory usage
    scale = 0.25  # Reduce image size by 75%
    rescaled_gray = rescale(img_gray, scale=scale, anti_aliasing=True)

    # 1b- Apply Canny edge detection on the rescaled image
    edges_rescaled = canny(rescaled_gray, sigma=2)

    # 1c- Define possible circle radii for outermost circle
    radius = (dish_diameter / 2.0)* scale
    low = (radius - radius * 0.04)
    high = (radius + radius * 0.04)
    hough_radii_rescaled_large = np.arange(low, high, 2)  # Increased range for larger circles
    hough_res_rescaled_large = hough_circle(edges_rescaled, hough_radii_rescaled_large)

    # 1d- Extract the most prominent circles
    accums_large, cx_large, cy_large, radii_large = hough_circle_peaks(
        hough_res_rescaled_large, hough_radii_rescaled_large, total_num_peaks=5
    )

    # 1e- Select the largest circle by radius
    largest_circle_idx = np.argmax(radii_large)
    cx_final, cy_final, radius_final = (
        cx_large[largest_circle_idx],
        cy_large[largest_circle_idx],
        radii_large[largest_circle_idx],
    )

    # 1f- Scale circle properties back to original size
    cx_corrected = cx_final / scale
    cy_corrected = cy_final / scale
    radius_corrected = radius_final / scale


    # 1g- Generate 8 Equidistant Points Around Perimeter
    theta_vals = np.linspace(0, 2 * np.pi, 9)[:-1]  # 8 points at 45-degree intervals
    perimeter_points = np.array([
        [cx_corrected + radius_corrected * np.cos(theta),
         cy_corrected + radius_corrected * np.sin(theta)]
        for theta in theta_vals
    ])

    # 1h- find brightest point along center point
    center_point = (cx_corrected, cy_corrected)
    adjusted_points = find_brightest_with_dark_neighbor(img_gray, perimeter_points, center_point, threshold=50, search_distance=5)
    if(len(adjusted_points) != len(perimeter_points)):
        adjusted_points = find_brightest_with_dark_neighbor(img_gray, perimeter_points, center_point, threshold=50, search_distance=30)

    adjusted_points_np = np.array(adjusted_points)

    # 1i- find adjusted centroid + radius
    initial_guess = [cx_corrected, cy_corrected, radius_corrected]
    new_params, _ = leastsq(circle_residuals, initial_guess, args=(adjusted_points_np,))
    cx_adjusted, cy_adjusted, radius_adjusted = new_params


    # Save circle detection picture
    if save_circle_detection:
        # # wrt. resized image
        # dummy, img_circle_detection = resize_image(img, pixel_threshold=pixel_threshold)
        img_circle_detection = rescale(img_bgr, scale=scale, anti_aliasing=True, channel_axis=-1)
        img_circle_detection = np.copy(img_circle_detection)



        circy, circx = circle_perimeter(int(cy_final),
                                        int(cx_final),
                                        int(radius_final),
                                        shape=img_circle_detection.shape)
        # Draw green perimeter
        img_circle_detection[circy, circx] = (0, 255, 51)
        img_circle_detection = np.clip(img_circle_detection, 0, 1)

        rescaled_image_with_circle_uint8 = img_as_ubyte(img_circle_detection)


        imsave(f'{output_path}/circle_detection_scaled.png', rescaled_image_with_circle_uint8[:, :, ::-1])

        # wrt. original image
        img_circle_detection = img_bgr.copy()
        # Draw red perimiter (after rescaling to normal)
        # circy, circx = circle_perimeter(int(cy_corrected),
        #                                 int(cx_corrected),
        #                                 int(radius_corrected),
        #                                 shape=img_circle_detection.shape)

        # img_circle_detection[circy, circx] = (255, 0, 0)

        # Draw green perimeter ()
        circy, circx = circle_perimeter(int(cy_adjusted),
                                        int(cx_adjusted),
                                        int(radius_adjusted),
                                        shape=img_circle_detection.shape)

        img_circle_detection[circy, circx] = (0, 255, 51)

        # Add tickness to perimiter
        # for i in range(1, 4):
        #     img_circle_detection[circy-i, circx] = (0, 255, 51)
        #     img_circle_detection[circy+i, circx] = (0, 255, 51)
        #     img_circle_detection[circy, circx-i] = (0, 255, 51)
        #     img_circle_detection[circy, circx+i] = (0, 255, 51)


        # Draw adjusted points
        for point in adjusted_points_np:
            img_circle_detection[int(point[1]), int(point[0])] = (0, 0, 255)

        imsave(f'{output_path}/circle_detection_original.png', img_circle_detection[:, :, ::-1])

    # 1d- Scale centroid and radius back to original image
    centroid = (cy_adjusted, cx_adjusted)
    radius = radius_adjusted
    radius_offset = radius - dish_offset   # apply offset


    # 1e- Create circular masks
    circular_mask = create_circlular_mask(img_gray.shape[::-1], centroid[::-1], radius_offset)
    circular_mask_threshold = create_circlular_mask(img_gray.shape[::-1], centroid[::-1], radius_offset -8)

    # 1f- Mask original image
    idx = (circular_mask== False)
    img_masked = np.copy(img_bgr)
    img_masked[idx] = 0; # black

    # 1g- Crop image
    x_min = int(centroid[0]-radius_offset)
    x_max = int(centroid[0]+radius_offset)
    y_min = int(centroid[1]-radius_offset)
    y_max = int(centroid[1]+radius_offset)

    if(x_min < 0):
        x_min = 0
    if(x_max > img_masked.shape[0]):
        x_max = img_masked.shape[0]
    if(y_min < 0):
        y_min = 0
    if(y_max > img_masked.shape[1]):
        y_max = img_masked.shape[1]

    img_cropped = img_masked[x_min:x_max, y_min:y_max]
    img_original_cropped = img_bgr[x_min:x_max, y_min:y_max]
    circular_mask_threshold = circular_mask_threshold[x_min:x_max, y_min:y_max]

    # 2- Convert to grayscale
    img_gray = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)

    # 3- Blur image
    img_blur = cv.GaussianBlur(img_gray, (7, 7), 0)

    # 4- Adaptive threshold
    img_threshold = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 9, 2)

    # 5- Remove contour artifacts generated from adaptive threshold
    img_mask_artifacts = img_threshold.copy()
    idx = (circular_mask_threshold== False)
    img_mask_artifacts[idx] = 0; # black


    # 6- Closing - dilation followed by erosion
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    closing = cv.morphologyEx(img_mask_artifacts, cv.MORPH_CLOSE,kernel_close, iterations = 1)

    # 7- Opening - erosion followed by dilation
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN,kernel_open, iterations = 1)

    # Segmentation for separating different objects in an image
    # 8a- Generate the markers as local maxima of the distance to the background
    img_distance = np.empty(opening.shape)
    ndi.distance_transform_edt(opening, distances=img_distance)

    # Segment image using watershed technique
    print('Computing watershed...')
    coords = peak_local_max(img_distance, footprint=np.ones((3, 3)), labels=opening)
    img_peak_mask = np.zeros(img_distance.shape, dtype=bool)
    img_peak_mask[tuple(coords.T)] = True

    markers = ndi.label(img_peak_mask)[0]

    # 8b- Watershed
    img_labels = watershed(img_distance, markers, mask=opening, connectivity=1, compactness=0)

    # Generate metrics from labels
    # 9a- Add extra properties, some function must be populated afterwards
    print('Computing region properties...')
    extra_callbacks = (compactness_cb,
                       nn_collision_distance_cb,
                       nn_centroid_distance_cb,
                       cell_quality_cb,
                       discarded_cb,
                       discarded_description_cb,
                       axes_closness_cb)

    # 9b- Measure properties of labelled image regions
    properties = regionprops(img_labels, intensity_image=img_gray, extra_properties=extra_callbacks)
    print(f'Region properties = {len(properties)}')

    # 9c- Compute compactness
    for p in properties:
        # avoid division by zero
        if(p.perimeter == 0.0):
            p.compactness = 0.0
        else:
            p.compactness = compute_compactness(p.area, p.perimeter)

    # 9d- Remove every properties that have a perimeter of 0
    properties[:] = [p for p in properties if p["perimeter"] > 0.0]
    print(f'Region properties, after removing small objects = {len(properties)}')

    # 9e- Compute axes_closness
    for p in properties:
        # avoid division by zero
        if p.axis_major_length == 0.0 or p.axis_minor_length == 0:
            p.axes_closness = 0.0
        else:
            p.axes_closness = compute_axes_closness(p.axis_major_length, p.axis_minor_length)

    # 9f- Find the nearest neighbors with ckDTree
    print('Computing distance to nearest neighboring cells...')

    centroids = [p["centroid"] for p in properties]
    tree = cKDTree(centroids)
    k = nn_query_size
    if(k > len(centroids)):
        k = len(centroids)

    for i in range(0, len(centroids)):
        centroid = centroids[i]
        dd, ii = tree.query(centroid, k)

        p = properties[i]
        p.nn_centroid_distance = dd[1]
        radius = p.equivalent_diameter_area/2.0

        # compute collision distance
        prev_nn_diameter = float('-inf')
        prev_collision_distance = float('+inf')

        for index in range(1, len(ii)):
            pnn = properties[ii[index]]
            nn_diameter = pnn.equivalent_diameter_area

            # only compute if the radius of the cell is greater then the previous
            # cell, because the centroid distance will be greater, so a greater
            # radius must be achieved so that the collision distance may shrink
            if  nn_diameter > prev_nn_diameter:
                prev_nn_diameter = nn_diameter
                nn_radius = nn_diameter / 2.0

                collision_distance = dd[index] - (radius + nn_radius)

                if(collision_distance < prev_collision_distance):
                    prev_collision_distance = collision_distance
                    p.nn_collision_distance = collision_distance

    # 9g- Compute cell_quality metric an discard cell's based on user threshold
    max_nn_collision_distance = max(p["nn_collision_distance"] for p in properties if p["compactness"] >= 0.2 and p["nn_collision_distance"] >= 0)
    max_area = max(p["area"] for p in properties if p["compactness"] >= 0.2)

    quality_metrics = np.empty(len(properties), dtype=object)
    for i in range(0, len(properties)):
        p = properties[i]

        # normalize
        n_area = p.area / max_area
        n_nn_collision_distance = p.nn_collision_distance / max_nn_collision_distance

        # clamp compactness to 1
        n_compactness = p.compactness
        if(n_compactness > 1.0):
           n_compactness =1.0

        # invert eccentricity ratio
        n_eccentricity = 1.0 - p.eccentricity

        metrics_used = 0
        if(weight_area):
            metrics_used += 1
        if(weight_compactness):
            metrics_used += 1
        if(weight_eccentricity):
            metrics_used += 1
        if(weight_nn_collision_distance):
            metrics_used += 1
        if(weight_solidity):
            metrics_used += 1

        if( not metrics_used):
            print("Metrics weights are all zeroes")
            sys.exit(1)

        # compute quality metric
        cell_quality = (weight_area  * n_area +
                        weight_compactness  * n_compactness +
                        weight_eccentricity * n_eccentricity +
                        weight_nn_collision_distance  * n_nn_collision_distance +
                        weight_solidity  * p.solidity) / metrics_used

        # discard cells
        if(cell_min_diameter and p.equivalent_diameter < cell_min_diameter):
            p.discarded = True
            p.discarded_description += f'Cell equivalent diameter is lower then the requested threshold: {p.equivalent_diameter:.2f} < {cell_min_diameter:.2f}\n'
            cell_quality = 0.0
        if(cell_max_diameter and p.equivalent_diameter > cell_max_diameter):
            p.discarded = True
            p.discarded_description += f'Cell equivalent diameter is higher then the requested threshold: {p.equivalent_diameter:.2f} > {cell_max_diameter:.2f}\n'
            cell_quality = 0.0
        if(cell_min_compactness and p.compactness < cell_min_compactness):
            p.discarded = True
            p.discarded_description += f'Cell compactness is lower then the requested threshold: {p.compactness:.2f} < {cell_min_compactness:.2f}\n'
            cell_quality = 0.0
        if(cell_min_solidity and p.solidity < cell_min_solidity):
            p.discarded = True
            p.discarded_description += f'Cell solidity is lower then the requested threshold: {p.solidity:.2f} < {cell_min_solidity:.2f}\n'
            cell_quality = 0.0
        if(cell_max_eccentricity and p.eccentricity > cell_max_eccentricity):
            p.discarded = True
            p.discarded_description += f'Cell eccentricity is higher then the requested threshold: {p.eccentricity:.2f} > {cell_max_eccentricity:.2f}\n'
            cell_quality = 0.0
        if(p.nn_collision_distance < 0): # in collision
            p.discarded = True
            p.discarded_description += f'Cell is in collision. Distance: {p.nn_collision_distance:.2f}\n'
            cell_quality = 0.0


        quality_metrics[i] = (cell_quality, i)
        p.cell_quality = cell_quality


    # remove outliers wrt. diameter of non discarded cell's
    if(std_weight_diameter):
        diameter_std = statistics.pstdev(p["equivalent_diameter"] for p in properties if p.cell_quality > 0.0)
        diameter_mean = statistics.mean(p["equivalent_diameter"] for p in properties if p.cell_quality > 0.0)

        for i in range(0, len(properties)):
            p = properties[i]
            if (p.equivalent_diameter < diameter_mean - std_weight_diameter* diameter_std):
                p.discarded = True
                p.discarded_description += f'Cell diameter is outside the requested std distribution: {p.equivalent_diameter:.2f} < {(diameter_mean - std_weight_diameter* diameter_std):.2f},\nwhere mean = {diameter_mean:.2f}, sigma = {std_weight_diameter:.2f}, std = {diameter_std:.2f}\n'
                # print(f'label {p.label} discarded due to being below std')
                p.cell_quality = 0.0
                quality_metrics[i] = (p.cell_quality, i)
            elif(p.equivalent_diameter > diameter_mean + std_weight_diameter* diameter_std):
                p.discarded = True
                p.discarded_description += f'Cell diameter is outside the requested std distribution: {p.equivalent_diameter:.2f} > {(diameter_mean + std_weight_diameter* diameter_std):.2f},\nwhere mean = {diameter_mean:.2f}, sigma = {std_weight_diameter:.2f}, std = {diameter_std:.2f}\n'
                # print(f'label {p.label} discarded due to being below std')
                p.cell_quality = 0.0
                quality_metrics[i] = (p.cell_quality, i)



    # 10- sort by best cell_quality metrics (higher is better)
    # TODO itemgetter might be faster then a lambda
    # https://stackoverflow.com/a/10695158
    reverse_metrics = sorted(quality_metrics, key=lambda x: x[0], reverse=True)

    # 11- Reduce list to requested number of cells
    reverse_metrics_slice = reverse_metrics
    if(max_cells and len(reverse_metrics) > max_cells):
        reverse_metrics_slice = reverse_metrics[0:max_cells]


    # Save cell annotation
    ax_annotation = None
    if save_cell_annotation:
        if len(reverse_metrics_slice) != 0:
            result_img = img_original_cropped.copy()[:, :, ::-1]

            dpi_value = 100
            height, width = result_img.shape[:2]

            fig, ax = plt.subplots()
            ax.imshow(result_img)
            ax.axis('off')

            # circle up best matches
            index  = 1
            for metric in reverse_metrics_slice:
                p = properties[metric[1]]

                diameter = p["equivalent_diameter_area"]
                centroid_local = [p.centroid[0], p.centroid[1]]

                point = (0,0)
                point = (centroid_local[1], centroid_local[0])

                radius = diameter/2.0
                circle = plt.Circle(point, radius=radius, fc='none', color='red')
                ax.add_patch(circle)
                ax.annotate(index, xy=(point[0]+radius, point[1]-radius), color='red')
                index += 1

            # Set exact figure size to match original dimensions
            fig.set_size_inches(width / dpi_value, height / dpi_value)

            # Ensure there are no margins by setting tight layout and removing axis padding
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # Set transparent background
            fig.patch.set_alpha(0)

            plt.savefig(f'{output_path}/annotated_cell.png', bbox_inches='tight', pad_inches=0, dpi=dpi_value, transparent=True)

    # Save segementation process
    if save_segmentation_process:
        layout = [
            ["A", "B", "C",],
            ["D", "E", "F",],
            ["G", "H", "I",],
        ]

        fig, axd = plt.subplot_mosaic(layout, constrained_layout=True, dpi=300)


        axd['A'].imshow(img_original_cropped[:, :, ::-1])
        axd['A'].set_title('Crop')
        axd['A'].set_axis_off()

        axd['B'].imshow(img_gray, cmap=plt.cm.gray)
        axd['B'].set_title('Mask')
        axd['B'].set_axis_off()

        axd['C'].imshow(img_threshold, cmap=plt.cm.gray)
        axd['C'].set_title('Threshold')
        axd['C'].set_axis_off()

        axd['D'].imshow(img_mask_artifacts, cmap=plt.cm.gray)
        axd['D'].set_title('Filter')
        axd['D'].set_axis_off()

        axd['E'].imshow(closing, cmap=plt.cm.gray)
        axd['E'].set_title('Closing')
        axd['E'].set_axis_off()

        axd['F'].imshow(opening, cmap=plt.cm.gray)
        axd['F'].set_title('Opening')
        axd['F'].set_axis_off()

        axd['G'].imshow(img_distance, cmap=plt.cm.gray)
        axd['G'].set_title('Distance')
        axd['G'].set_axis_off()

        axd['H'].imshow(img_labels, cmap=plt.cm.nipy_spectral)
        axd['H'].set_title('Watershed')
        axd['H'].set_axis_off()

        if ax_annotation:
            print("OK")
            axd['I'] = ax_annotation
            axd['I'].set_title('Best Cell\'s')
            axd['I'].set_axis_off()

        plt.tight_layout()
        plt.savefig(f"{output_path}/segmentation.png")


        # save as images
        imsave(f'{output_path}/crop.png', img_original_cropped[:, :, ::-1])
        imsave(f'{output_path}/mask.png', img_gray)
        imsave(f'{output_path}/threshold.png', img_threshold)
        imsave(f'{output_path}/filter.png', img_mask_artifacts)
        imsave(f'{output_path}/closing.png', closing)
        imsave(f'{output_path}/opening.png', opening)

        # Normalize and convert to uint8
        dist_normalized = img_distance / np.max(img_distance)
        dist_uint8 = img_as_ubyte(dist_normalized)
        imsave(f'{output_path}/distance.png', dist_uint8)

        plt.imsave(f'{output_path}/watershed.png', img_labels, cmap='nipy_spectral')


    # Generate excel properties sheet
    if save_properties:
        # select properties included in the table
        prop_names = ('label',
                      'cell_quality',    # custom
                      'compactness',     # custom
                      'nn_centroid_distance', # custom
                      'nn_collision_distance', # custom
                      'discarded', # custom
                      'discarded_description', # custom
                      'axes_closness',   # custom
                      'area',
                      'area_bbox',
                      'area_convex',
                      'area_filled',
                      'axis_major_length',
                      'axis_minor_length',
                      'bbox',
                      'centroid',
                      'centroid_local',
                      'centroid_weighted',
                      'centroid_weighted_local',
                      'coords',
                      'coords_scaled',
                      'eccentricity',
                      'equivalent_diameter_area',
                      'euler_number',
                      'extent',
                      'feret_diameter_max',
                      'image',
                      'image_convex',
                      'image_filled',
                      'image_intensity',
                      'inertia_tensor',
                      'inertia_tensor_eigvals',
                      'intensity_max',
                      'intensity_mean',
                      'intensity_min',
                      # 'intensity_std', # requires scikit-image 0.24.0
                      'label',
                      'moments',
                      'moments_central',
                      'moments_hu',
                      'moments_normalized',
                      'moments_weighted',
                      'moments_weighted_central',
                      'moments_weighted_hu',
                      'moments_weighted_normalized',
                      'num_pixels',
                      'orientation',
                      'perimeter',
                      'perimeter_crofton',
                      'slice',
                      'solidity',)

        # Generate dictionary of properties
        props_dict = regionprops_to_dict(properties, prop_names)

        # Find the max length of string metrics, e.g. with 'discarded_description'
        max_description_size = 0
        for p in properties:
            description_size = len(p.discarded_description)
            if(description_size > max_description_size):
                max_description_size = description_size

        # Recreate string metrics with a fixed unicode string
        props_dict["discarded_description"] = np.empty(len(properties), dtype=f'<U{max_description_size}')
        for i in range(len(properties)):
            props_dict["discarded_description"][i] = properties[i].discarded_description

        # Convert to DataFrame
        df = pd.DataFrame(props_dict)
        df.to_excel(f'{output_path}/region_properties.xlsx', index=False)


    # Plot interactive region properties
    if plot_interactive_properties:
        property_names = ['area',
                          'eccentricity',
                          'perimeter',
                          'solidity',
                          'compactness',
                          'axes_closness',
                          'equivalent_diameter',
                          'nn_collision_distance',
                          'cell_quality',
                          'discarded',
                          'discarded_description']

        print(f"Plotting region properties ({len(properties)}) interactively, this may take some time...")
        plot_region_roperties(img_original_cropped, img_labels, properties, property_names)


    print("===")
    print("Completed")
    print("===")
