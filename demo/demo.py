#!/usr/bin/env python3
from colony_morphology.geometry import *
from colony_morphology.plotting import plot_bboxes, plot_region_roperties
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.measure import regionprops, regionprops_table, label
from skimage.segmentation import watershed


if __name__=="__main__":
    # Parameters
    min_size_small_holes = 50
    min_size_small_objects = 50
    number_of_cells_to_pick = 15


    plot_segmentation_process = True
    plot_cell_annotation = True
    plot_interactive_properties = True

    # Read image
    img = cv.imread('/home/captain-yoshi/ws/Mimik/colony-morphology/dataset/sure_foreground.png')
    assert img is not None, "file could not be read, check with os.path.exists()"

    # Convert to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Remove small objects + Fill small holes
    img_bw = img_gray > 0
    img_rm_small_objects = morphology.remove_small_objects(img_bw, min_size_small_objects)
    img_fill_holes = morphology.remove_small_holes(img_rm_small_objects, min_size_small_holes)

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    img_distance = np.empty(img_fill_holes.shape)
    ndi.distance_transform_edt(img_fill_holes, distances=img_distance)

    # Segment image using watershed technique
    print('Computing watershed')
    coords = peak_local_max(img_distance, footprint=np.ones((3, 3)), labels=img_gray)
    img_peak_mask = np.zeros(img_distance.shape, dtype=bool)
    img_peak_mask[tuple(coords.T)] = True

    markers = ndi.label(img_peak_mask)[0]
    img_labels = watershed(-img_distance, markers, mask=img_gray, connectivity=1, compactness=0)

    # Retrieve metric from labels
    # Do not remove small objects as it will be required to remove overlapping colonies
    print('Computing region properties')
    properties = regionprops(img_labels, intensity_image=None)
    print(f'properties size={len(properties)}')

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

    # Compute metric for best colonies
    # Defined as:
    #     max distance from nn * area * eccentricity^-1
    quality_metrics = []
    for p in properties_wo_overlap:
        nn = min_distance_nn[p.label - 1]
        nn_distance = nn[1]
        quality_metrics.append((p.area * nn_distance * (1.0/p.eccentricity), p.label))

    quality_metrics.sort(reverse = True)



    # Print image segmentation process
    if(plot_segmentation_process):
        fig, axes = plt.subplots(ncols=4, nrows=3, figsize=(9, 9), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(img)
        ax[0].set_title('Input image')
        ax[1].imshow(img_gray, cmap=plt.cm.gray)
        ax[1].set_title('Grayscale')
        ax[2].imshow(img_rm_small_objects, cmap=plt.cm.gray)
        ax[2].set_title(f'Small objects removed min size={min_size_small_objects}')
        ax[3].imshow(img_fill_holes, cmap=plt.cm.gray)
        ax[3].set_title(f'Small holes removed min size={min_size_small_holes}')

        ax[4].imshow(img_distance, cmap=plt.cm.gray)
        ax[4].set_title('Distance transform')
        ax[5].imshow(-img_distance, cmap=plt.cm.gray)
        ax[5].set_title('Distance transform negative')

        ax[6].imshow(img_labels, cmap=plt.cm.nipy_spectral)
        ax[6].set_title('Watershed segmentation')

        ax[7].imshow(img)
        ax[7].set_title(f'Cell\'s Centroid N={len(properties)}')
        # add X marks on centroids
        for p in properties:
            ax[7].scatter(p.centroid[1], p.centroid[0], s=20, c='red', marker='x', linewidth=1)

        ax[8].imshow(img)
        ax[8].set_title(f'Overlapping cell\'s removed N={len(properties_wo_overlap)}')
        # add X marks on centroids
        for p in properties_wo_overlap:
            ax[8].scatter(p.centroid[1], p.centroid[0], s=20, c='red', marker='x', linewidth=1)

        ax[9].imshow(img)
        ax[9].set_title(f'Best colonies to pick')

        ax[9].patches
        # circle up best matches
        index = 1
        for p in properties_wo_overlap:
            point = (p.centroid[1], p.centroid[0])
            radius = p.equivalent_diameter_area/2.0 + 5

            circle = plt.Circle(point, radius=radius, fc='none', color='red')
            ax[9].add_patch(circle)
            ax[9].annotate(index, xy=(point[0]+radius, point[1]+radius), color='red')
            index += 1

            # ax[9].plot(point[0], point[1], 'o', ms=p.equivalent_diameter_area, mfc=None, mew=2)
            #break
            # ax[9].add_pathc(p.centroid[1], p.centroid[0], s=20, c='red', marker='x', linewidth=1)



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


        # index = 1
        # for p in properties_wo_overlap:
        #     point = (p.centroid[1], p.centroid[0])
        #     radius = p.equivalent_diameter_area/2.0 + 5

        #     circle = plt.Circle(point, radius=radius, fc='none', color='red')
        #     ax.add_patch(circle)
        #     ax.annotate(index, xy=(point[0]+radius, point[1]-radius), color='red')
        #     index += 1

        plt.tight_layout()
        plt.show()


    # Plot region properties interactively
    if (plot_interactive_properties):
        property_names = ['area', 'eccentricity', 'perimeter']

        print("Generating region properties interactively plot, this may take some time...")
        plot_region_roperties(img, img_labels, properties_wo_overlap, property_names)
