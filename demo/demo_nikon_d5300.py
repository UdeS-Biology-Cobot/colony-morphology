#!/usr/bin/env python3
from colony_morphology.geometry import *
from colony_morphology.plotting import plot_bboxes, plot_region_roperties, image_resize
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.measure import regionprops, regionprops_table, label
from skimage.segmentation import watershed


import os

def find_petri_dish(image):
    max_pixels = 1280*1280
    median_blur_order = 5
    minmax_radius = (1500, 2000)

    # resize image to speedup algorithm
    img_resize = image
    scale = 1.0
    max_x = image.shape[1]
    max_y = image.shape[0]
    if(image.size > max_pixels):
        pixels = max_x * max_y
        scale = 1.0 / (pixels / max_pixels);
        point = (int(scale * max_x), int(scale * max_y))

        img_resize = cv.resize(image, point, interpolation= cv.INTER_LINEAR)


    # Convert to grayscale
    img_gray = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)

    # Blur grayscale image
    img_blur = cv.medianBlur(img_gray, median_blur_order)

    # Find circles is the image with Hough Circle Transform
    # The algorithm returns a list of (x, y, radius) where (x, y) is center
    minmax_radius_scaled = (int(minmax_radius[0]*scale), int(minmax_radius[1]*scale))


    circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 2, \
                 1, minRadius=minmax_radius_scaled[0], maxRadius=minmax_radius_scaled[1])

    if not isinstance(circles, type(np.empty(0))):
        print("Error - no circle found")
        raise Exception('Error - no circle found')



    circles = np.uint16(np.around(circles))
    smallest_radius = float('inf')
    img_alternate = img_resize.copy()
    circle = None

    for i in circles[0,:]:
        if(i[2] < smallest_radius):
            smallest_radius = i[2]
            circle = i

        # draw the outer circle
        cv.circle(img_resize,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(img_resize,(i[0],i[1]),2,(0,0,255),3)

    cv.imshow('detected circles',img_resize)
    cv.waitKey(0)


    cv.circle(img_alternate,(circle[0],circle[1]),circle[2],(0,255,0),2)
        # draw the center of the circle
    cv.circle(img_alternate,(circle[0],circle[1]),2,(0,0,255),3)

    cv.imshow('smallest radiuscircles',img_alternate)
    cv.waitKey(0)

    # Make cicular mask
    inv_scale = 1/scale
    c_x = np.round(inv_scale*circle[0])
    c_y = np.round(inv_scale*circle[1])

    nx = np.linspace(-c_x, max_x - c_x - 1, max_x)
    ny = np.linspace(-c_y, max_y - c_y - 1, max_y)
    mesh_x, mesh_y = np.meshgrid(nx, ny)
    c_mask = mesh_x ** 2 + mesh_y ** 2 <= (0.8*inv_scale*circle[2]) ** 2

    # Apply circular mask
    idx = (c_mask== False)
    img_masked = np.copy(image)
    img_masked[idx] = 0;
    cv.imshow("BW2", image_resize(img_masked, height=1280))
    cv.waitKey(0)
    cv.destroyAllWindows()


    return img_masked



if __name__=="__main__":
    # Parameters
    min_size_small_holes = 50
    min_size_small_objects = 50
    number_of_cells_to_pick = 15


    plot_segmentation_process = True
    plot_cell_annotation = True
    plot_interactive_properties = True

    # Read image
    absolute_path = os.path.join(os.getcwd(), "dataset/nikon_d5300_1.jpg")

    img = cv.imread(absolute_path)
    assert img is not None, "file could not be read, check with os.path.exists()"

    # # Mask image to contain only the petri dish
    img_dish = find_petri_dish(img)
    # raise Exception("Sorry, no numbers below zero")

    # Convert to grayscale
    img_gray = cv.cvtColor(img_dish, cv.COLOR_BGR2GRAY)
    # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    # Blur image
    img_blur = cv.GaussianBlur(img_gray, (7, 7), 0)


    # Apply threshold
    (thresh, img_bw) = cv.threshold(img_blur, 175, 255, cv.THRESH_BINARY)
    # (thresh, img_bw) = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    # img_bw = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 10, 10)


    # Noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(img_bw,cv.MORPH_OPEN,kernel, iterations = 2)

    # Sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    distance = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(distance,0.1*distance.max(),255,0)

    # Convert to int
    sure_fg = np.uint8(sure_fg)

    # Remove small objects + Fill small holes
    # It turns out we need the small objects to actually compute the min distance to nearest neighboring cells
    # img_rm_small_objects = morphology.remove_small_objects(sure_fg, min_size_small_objects)
    # img_fill_holes = morphology.remove_small_holes(img_, min_size_small_holes)

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    img_distance = np.empty(sure_fg.shape)
    ndi.distance_transform_edt(sure_fg, distances=img_distance)

    # Segment image using watershed technique
    print('Computing watershed')
    coords = peak_local_max(img_distance, footprint=np.ones((3, 3)), labels=sure_fg)
    img_peak_mask = np.zeros(img_distance.shape, dtype=bool)
    img_peak_mask[tuple(coords.T)] = True

    markers = ndi.label(img_peak_mask)[0]
    img_labels = watershed(-img_distance, markers, mask=sure_fg, connectivity=1, compactness=0)

    # Retrieve metric from labels
    # Do not remove small objects as it will be required to remove overlapping colonies
    print('Computing region properties')
    properties = regionprops(img_labels, intensity_image=None)
    print(f'Properties size = {len(properties)}')

    # Compute min distance to nearest neighboring cells from edge (~1minute 30 seconds)
    # TODO from centroids would be faster, but wold require to change the instersection removal...
    print('Computing distance to nearest neighboring cells, this may take some time...')
    polygons_dict = make_polygons_from_mask(img_labels)
    distances_dict = pairwise_polygon_distance(polygons_dict)

    min_distance_nn = []
    for p in properties:
        min_distance_nn.append(get_nn_distance(p.label, distances_dict))

    # for p in properties:
    #     print(f'#{p.label} in collision with #{get_nn_distance(p.label, distances_dict)[0]}, distance = {get_nn_distance(p.label, distances_dict)[1]}')

    # Remove intersecting cells by checking overlaps in bounding boxes
    properties_wo_overlap = properties[:]
    first_label_number = properties[0].label
    for i  in range(len(properties)):
        nn = min_distance_nn[i]
        nn_label = nn[0]
        nn_distance = nn[1]

        first_prop = properties[i]
        second_prop = properties[nn_label-1]

        # skip intersect if distance is zero
        if(nn_distance == 0.0):
            try:
                properties_wo_overlap.remove(first_prop)
                properties_wo_overlap.remove(second_prop)
            except ValueError:
                 pass  # do nothing
        if(intersects(first_prop.bbox, second_prop.bbox)):
            try:
                properties_wo_overlap.remove(first_prop)
                properties_wo_overlap.remove(second_prop)
            except ValueError:
                 pass  # do nothing

    print(f'Properties remaining = {len(properties_wo_overlap)}')


    # Compute metric for best colonies
    # Defined as:
    #     max distance from nn * area * (1 - eccentricity)
    quality_metrics = []
    for p in properties_wo_overlap:
        nn = min_distance_nn[p.label - 1]
        nn_distance = nn[1]
        quality_metrics.append((p.area * nn_distance * (1.0 - p.eccentricity), p.label))

    quality_metrics.sort(reverse = True)



    # Print image segmentation process
    if(plot_segmentation_process):
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

        ax[8].imshow(img)
        ax[8].set_title(f'Cell\'s Centroid N={len(properties)}')
        # add X marks on centroids
        for p in properties:
            ax[8].scatter(p.centroid[1], p.centroid[0], s=20, c='red', marker='x', linewidth=1)

        ax[9].imshow(img)
        ax[9].set_title(f'Overlapping cell\'s removed N={len(properties_wo_overlap)}')
        # add X marks on centroids
        for p in properties_wo_overlap:
            ax[9].scatter(p.centroid[1], p.centroid[0], s=20, c='red', marker='x', linewidth=1)

        ax[10].imshow(img)
        ax[10].set_title(f'Best colonies to pick')

        ax[10].patches
        # circle up best matches
        index = 1
        for p in properties_wo_overlap:
            point = (p.centroid[1], p.centroid[0])
            radius = p.equivalent_diameter_area/2.0 + 5

            circle = plt.Circle(point, radius=radius, fc='none', color='red')
            ax[10].add_patch(circle)
            ax[10].annotate(index, xy=(point[0]+radius, point[1]+radius), color='red')
            index += 1


        for a in ax:
            a.set_axis_off()

        plt.tight_layout()
        plt.show()

    # Annotate best cells to pick
    if (plot_cell_annotation):
        fig, ax = plt.subplots()

        ax.imshow(img)
        ax.set_title(f'Best colonies to pick')

        # circle up best matches
        index  = 1
        for metric in quality_metrics:
            p = properties[metric[1]-1]

            point = (p.centroid[1], p.centroid[0])
            radius = p.equivalent_diameter_area/2.0 + 5

            circle = plt.Circle(point, radius=radius, fc='none', color='red')
            ax.add_patch(circle)
            ax.annotate(index, xy=(point[0]+radius, point[1]-radius), color='red')
            
            if (index == number_of_cells_to_pick):
                break
            index += 1

        plt.tight_layout()
        plt.show()


    # Plot region properties interactively
    if (plot_interactive_properties):
        property_names = ['area', 'eccentricity', 'perimeter']

        print("Generating region properties interactively plot, this may take some time...")
        plot_region_roperties(img, img_labels, properties_wo_overlap, property_names)
        # plot_region_roperties(img, img_labels, properties, property_names)
