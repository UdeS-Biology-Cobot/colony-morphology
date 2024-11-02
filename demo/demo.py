#!/usr/bin/env python3
from colony_morphology.geometry import *
from colony_morphology.image_transform import *
from colony_morphology.plotting import plot_bboxes, plot_region_roperties
from colony_morphology.skimage_util import compactness, min_distance_nn, cell_quality
from colony_morphology.metric import compactness as compute_compactness

import argparse
import cv2 as cv
from io import BytesIO
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import color, io, morphology
from skimage.feature import peak_local_max
from skimage.filters import try_all_threshold, threshold_yen, threshold_otsu, threshold_local
from skimage.measure import regionprops, regionprops_table, label
from skimage.segmentation import watershed
from numpy.linalg import norm
import sys
import time

if __name__=="__main__":

    # Parsing
    parser = argparse.ArgumentParser(
        description="Script that finds best colonies to pick"
    )

    parser.add_argument("--image_path", required=True, type=str)
    parser.add_argument("--dish_radius", required=True, type=float)
    parser.add_argument("--dish_inner_offset", required=False, type=float, default=0)
    parser.add_argument("--cell_min_radius", required=True, type=int)
    parser.add_argument("--cell_max_radius", required=False, type=int, default=sys.maxsize)
    parser.add_argument("--number_of_cells", required=False, type=int, default=15)
    args = parser.parse_args()

    # Parameters
    image_path = args.image_path
    dish_radius = args.dish_radius
    dish_offset_radius =  args.dish_inner_offset
    cell_min_radius = args.cell_min_radius
    cell_max_radius = args.cell_max_radius
    number_of_cells_to_pick = args.number_of_cells

    absolute_path = image_path
    if (not os.path.isabs(image_path)):
        absolute_path = os.path.join(os.getcwd(), image_path)


    plot_petri_process = True
    plot_all_threshold = True
    plot_segmentation_process = False
    plot_cell_annotation = True
    plot_interactive_properties = False
    save_pictures = False

    cell_min_diameter = cell_min_radius * 2.0
    cell_max_diameter = cell_max_radius * 2.0

    matplotlib.rcParams['savefig.dpi'] = 300

    # Read image
    print(f'Reading image from: {absolute_path}')
    img = cv.imread(image_path)
    assert img is not None, "File could not be read, check with os.path.exists()"

    # Compute total algorithm processingtime, excluding plotting
    global_time = 0
    global_start = time.time()

    # Convert to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    # Mask image to contain only the petri dish
    # 1- resize image to speedup detection
    scale, img_resize = resize_image(img_gray, pixel_threshold=1280*1280)

    # 2- detect circle radius + centroid
    dish_regions = detect_area_by_canny(img_resize, radius=scale*dish_radius)

    region_prop = dish_regions[0]
    centroid = region_prop["centroid"]
    diameter = region_prop["equivalent_diameter_area"]

    centroid = tuple(c/scale for c in centroid)
    diameter/=scale
    diameter -= 2*dish_offset_radius

    # 3- create circular mask
    circular_mask = create_circlular_mask(img_gray.shape[::-1], centroid[::-1], diameter/2.0)

    # 4- mask orighinal image
    idx = (circular_mask== False)
    img_masked = np.copy(img)
    img_masked[idx] = 255; # make mask white to lower the size of fake region properties found

    diameter += 2*dish_offset_radius

    x_min = int(centroid[0]-diameter/2)
    x_max = int(centroid[0]+diameter/2)
    y_min = int(centroid[1]-diameter/2)
    y_max = int(centroid[1]+diameter/2)

    if(x_min < 0):
        x_min = 0
    if(x_max > img_masked.shape[0]):
        x_max = img_masked.shape[0]
    if(y_min < 0):
        y_min = 0
    if(y_max > img_masked.shape[1]):
        y_max = img_masked.shape[1]


    # 5- crop image
    img_cropped = img_masked[x_min:x_max, y_min:y_max]
    img_original_cropped = img[x_min:x_max, y_min:y_max]


    # Plot Petri process
    if (plot_petri_process):
        global_time += time.time() - global_start

        print(f'Plotting petri process...')
        start = time.time()

        fig, axes = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(img_resize, cmap=plt.cm.gray)
        ax[0].set_title(f'Image resized')
        ax[1].imshow(circular_mask, cmap=plt.cm.gray)
        ax[1].set_title(f'Circular mask')

        ax[2].imshow(img_masked, cmap=plt.cm.gray)
        ax[2].set_title(f'Image cropped')

        for a in ax:
            a.set_axis_off()

        plt.tight_layout()

        end = time.time()
        print(f'[{end-start:g} seconds]')

        plt.show()

    global_start = time.time()

    # Convert to grayscale
    img_gray = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)

    # Blur image
    img_blur = cv.GaussianBlur(img_gray, (7, 7), 0)


    # plot all threshold algorithm's
    if(plot_all_threshold):
        fig, ax = try_all_threshold(img_blur, figsize=(10, 8), verbose=False)
        plt.show()



    # Apply threshold
    # (thresh, img_bw) = cv.threshold(img_blur, 195, 255, cv.THRESH_BINARY_INV)
    # (thresh, img_bw) = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    # img_bw = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 5, 2)
    img_bw = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 9, 2)


    # fill smal holes
    # https://stackoverflow.com/a/10317883
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9))
    img_fill = cv.morphologyEx(img_bw,cv.MORPH_CLOSE,kernel)


    # Noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(img_fill,cv.MORPH_OPEN,kernel, iterations = 2)
    img_bw = opening

    # Noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(img_bw,cv.MORPH_OPEN,kernel, iterations = 2)

    # Sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    distance = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(distance,0.1*distance.max(),255,0)
    sure_fg = np.uint8(sure_fg)     # Convert to int

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    img_distance = np.empty(sure_fg.shape)
    ndi.distance_transform_edt(sure_fg, distances=img_distance)

    # Segment image using watershed technique
    print('Computing watershed...')
    start = time.time()

    coords = peak_local_max(img_distance, footprint=np.ones((3, 3)), labels=sure_fg)
    img_peak_mask = np.zeros(img_distance.shape, dtype=bool)
    img_peak_mask[tuple(coords.T)] = True

    markers = ndi.label(img_peak_mask)[0]
    img_labels = watershed(img_distance, markers, mask=sure_fg, connectivity=1, compactness=0)

    end = time.time()
    print(f'[{end-start:g} seconds]')
    print(f'    Created {len(img_labels)} labels')

    # Retrieve metric from labels
    print('Computing region properties...')
    start = time.time()

    # add extra properties, some function must be populated afterwards
    extra_callbacks = (compactness, min_distance_nn, cell_quality)

    properties = regionprops(img_labels, intensity_image=img_cropped, extra_properties=extra_callbacks)

    end = time.time()
    print(f'[{end-start:g} seconds]')
    print(f'    Region properties found = {len(properties)}')

    # Compute compactness
    for p in properties:
        p.compactness = compute_compactness(p.area, p.perimeter)


    # Find the nearest neighbors with ckDTree
    print('Computing distance to nearest neighboring cells...')
    start = time.time()

    centroids = [p["centroid"] for p in properties]
    tree = cKDTree(centroids)

    for i in range(0, len(centroids)):
        centroid = centroids[i]
        dd, ii = tree.query(centroid, 2)

        p = properties[i]
        p.min_distance_nn = dd[1]

    end = time.time()
    print(f'[{end-start:g} seconds]')


    # Compute metric for best colonies
    # Defined as:
    #     max distance from nn * area * (1 - eccentricity)
    quality_metrics = []
    quality_metrics = np.empty(len(properties), dtype=object)
    for i in range(0, len(properties)):
        p = properties[i]
        quality_metrics[i] = (p.area * p.min_distance_nn * (1.0 - p.eccentricity), p.label)

        # discard cells
        if(p.equivalent_diameter_area < cell_min_diameter or
           p.equivalent_diameter_area > cell_max_diameter):
            quality_metrics[i] = (0, p.label)

        p.cell_quality = quality_metrics[i][0]

    reverse_metrics = sorted(quality_metrics, key=lambda x: (-x[0], x[1]))


    # print algorithm process time, excluding plotting
    global_time += time.time() - global_start
    print(f'Pipeline processing took {global_time:g} seconds (excluding plotting)')

    # Print image segmentation process
    if(plot_segmentation_process):
        print(f'Plotting segmentation process...')
        start = time.time()


        fig, axes = plt.subplots(ncols=4, nrows=3, figsize=(9, 9), sharex=True, sharey=True)
        ax = axes.ravel()


        ax[0].imshow(img)
        ax[0].set_title('Input image')
        ax[1].imshow(img_gray, cmap=plt.cm.gray)
        ax[1].set_title('Grayscale')
        ax[2].imshow(img_bw, cmap=plt.cm.gray)
        ax[2].set_title(f'Threshold')
        ax[3].imshow(sure_bg, cmap=plt.cm.gray)
        ax[3].set_title(f'Sure Foreground')

        ax[4].imshow(sure_fg, cmap=plt.cm.gray)
        ax[4].set_title(f'Sure Background')
        ax[5].imshow(img_distance, cmap=plt.cm.gray)
        ax[5].set_title('Distance transform')
        ax[6].imshow(-img_distance, cmap=plt.cm.gray)
        ax[6].set_title('Distance transform negative')

        ax[7].imshow(img_labels, cmap=plt.cm.nipy_spectral)
        ax[7].set_title('Watershed segmentation')

        ax[8].imshow(img_original_cropped)
        ax[8].set_title(f'Cell\'s Centroid N={len(properties)}')
        # add X marks on centroids
        for p in properties:
            ax[8].scatter(p.centroid[1], p.centroid[0], s=20, c='red', marker='x', linewidth=1)

        ax[9].imshow(img_original_cropped)
        ax[9].set_title(f'Overlapping cell\'s removed N={len(properties)}')
        # add X marks on centroids
        for p in properties:
            ax[9].scatter(p.centroid[1], p.centroid[0], s=20, c='red', marker='x', linewidth=1)

        ax[10].imshow(img_original_cropped)
        ax[10].set_title(f'Best colonies to pick')

        ax[10].patches
        # circle up best matches
        index = 1
        for p in properties:
            point = (p.centroid[1], p.centroid[0])
            radius = p.equivalent_diameter_area/2.0 + 5

            circle = plt.Circle(point, radius=radius, fc='none', color='red')
            ax[10].add_patch(circle)
            ax[10].annotate(index, xy=(point[0]+radius, point[1]+radius), color='red')
            index += 1


        for a in ax:
            a.set_axis_off()

        plt.tight_layout()

        end = time.time()
        print(f'[{end-start:g} seconds]')

        plt.show()



    # Annotate best cells to pick
    imgdata = BytesIO()
    if (plot_cell_annotation):
        fig, ax = plt.subplots()

        ax.imshow(img_original_cropped)
        ax.set_title(f'Best colonies to pick')

        # circle up best matches
        index  = 1
        for metric in reverse_metrics:
            p = properties[metric[1]-1]

            point = (p.centroid[1], p.centroid[0])
            radius = p.equivalent_diameter_area/2.0 + 5

            circle = plt.Circle(point, radius=radius, fc='none', color='red')
            ax.add_patch(circle)
            ax.annotate(index, xy=(point[0]+radius, point[1]-radius), color='red')

            if (index == number_of_cells_to_pick):
                break
            index += 1


        # plt.figure(dpi=1200)
        if(save_pictures):
            plt.savefig(imgdata, format='png')
            imgdata.seek(0) # rewind the data


        plt.tight_layout()
        plt.show()


    # Plot region properties interactively
    if (plot_interactive_properties):
        property_names = ['area', 'eccentricity', 'perimeter', 'solidity', 'compactness', 'min_distance_nn', 'cell_quality']

        print("Generating region properties interactively plot, this may take some time...")
        plot_region_roperties(img_original_cropped, img_labels, properties, property_names)



    if(save_pictures):

        path = os.path.join(os.getcwd(), "result/")
        path += time.strftime("%Y%m%d-%H%M%S") #avoid name clash

        # Create the directory
        try:
            os.mkdir(path)
            print(f'Saving images to: {path}')
        except FileExistsError:
            print(f"Directory '{path}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

        io.imsave(path + '/cropped.png', img_cropped)
        with open(path + '/annotated.png', "wb") as f:
            f.write(imgdata.read())
