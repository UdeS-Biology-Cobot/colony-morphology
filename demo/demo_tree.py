#!/usr/bin/env python3
from colony_morphology.geometry import *
from colony_morphology.image_transform import *
from colony_morphology.plotting import plot_bboxes, plot_region_roperties, image_resize
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import color, io, morphology
from skimage.feature import peak_local_max
from skimage.measure import regionprops, regionprops_table, label
from skimage.segmentation import watershed
from numpy.linalg import norm
import time

if __name__=="__main__":
    # Parameters
    dish_radius = 877 # 877
    dish_offset_radius =  100 #100
    min_size_small_holes = 50
    min_size_small_objects = 50
    number_of_cells_to_pick = 15


    plot_segmentation_process = True
    plot_cell_annotation = True
    plot_interactive_properties = True

    # Read image
    img = cv.imread('/home/captain-yoshi/ws/Mimik/colony-morphology/dataset/ref_backlight_hd1900.jpg')
    assert img is not None, "file could not be read, check with os.path.exists()"

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
    print(f'dimaeter = {diameter}')
    diameter -= 2*dish_offset_radius

    print(f'dimater ={diameter}')
    # 3- create circular mask
    circular_mask = create_circlular_mask(img_gray.shape[::-1], centroid[::-1], diameter/2.0)

    # 4- mask orighinal image
    idx = (circular_mask== False)
    img_masked = np.copy(img)
    img_masked[idx] = 255; # make mask white to lower the size of fake region properties found
    cv.imshow("Original image masked (White)", image_resize(img_masked, height=720));
    cv.waitKey(0)
    cv.destroyAllWindows()
    

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


    # Crop image
    img_cropped = img_masked[x_min:x_max, y_min:y_max]
    cv.imshow("LOL", image_resize(img_cropped, height=720));
    cv.waitKey(0)
    cv.destroyAllWindows()


    img_original_cropped = img[x_min:x_max, y_min:y_max]



    # Convert to grayscale
    img_gray = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)

    # Blur image
    img_blur = cv.GaussianBlur(img_gray, (7, 7), 0)

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


    cv.imshow("threshold", image_resize(img_bw, width=600))
    cv.imshow("fill holes", image_resize(img_fill, width=600))
    cv.imshow("removing noise", image_resize(opening, width=600))
    cv.waitKey(0)
    cv.destroyAllWindows()

    img_bw = opening

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
    img_labels = watershed(img_distance, markers, mask=sure_fg, connectivity=1, compactness=0)
    # img_labels = watershed(img_bw, markers,  connectivity=1, compactness=0)
    print(f'img_l;abels size = {len(img_labels)}')








    # Retrieve metric from labels
    # Do not remove small objects as it will be required to remove overlapping colonies
    print('Computing region properties')
    properties = regionprops(img_labels, intensity_image=img_cropped)
    print(f'Properties size = {len(properties)}')




    # find the neighbors with ckDTree

    print('Computing distance to nearest neighboring cells, this may take some time...')
    start = time.time()
    centroids = [p["centroid"] for p in properties]
    centroid_tree = cKDTree(centroids)
    properties_wo_overlap = properties[:]


    min_distance_nn = np.empty(len(properties), dtype=object)

    for i in range(0, len(centroids)):
        centroid = centroids[i]
        # prop = properties[i]

        dd, ii = centroid_tree.query(centroid, 2)

        min_distance_nn[i] = dd[1]

        # dd, ii = centroid_tree.query_ball_tree(centroid, 20)
        # ii = centroid_tree.query_ball_point(centroid, 20)

        # if(len(ii) < 2):
        #     min_distance_nn.append(norm(centroid-))
        #     continue

        # min_distance_nn.append(dd[1])

        # print(ii)
        # nn_prop = properties[ii[1]]
        # # print(dd)
        # print(f'label{prop.label} closest neighbor is {properties[ii[1]].label}')

        # if (len(dd) < 2):
        #     continue
        # if (dd[1] == 0.0):
        #     try:
        #         properties_wo_overlap.remove(prop)
        #         properties_wo_overlap.remove(nn_prop)
        #     except ValueError:
        #          pass  # do nothing
        #     continue

        # if(intersects(prop["bbox"], nn_prop["bbox"])):
        #     try:
        #         properties_wo_overlap.remove(prop)
        #         properties_wo_overlap.remove(nn_prop)
        #     except ValueError:
        #          pass  # do nothing

        #     continue
        # print("Mario-----------------------------------")
        # min_distance_nn.append(dd[1])


    # print("finished !!!")
    end = time.time()
    print(end - start)



    # Remove intersecting cells by checking overlaps in bounding boxes
    properties_wo_overlap = properties[:]
    # first_label_number = properties[0].label
    # for i  in range(len(properties)):

    #     nn = []
    #     try:
    #         nn = min_distance_nn[i]
    #     except IndexError:
    #         print(f'i = {i}, len={len(properties)}')
    #         continue

    #     nn_label = nn[0]
    #     nn_distance = nn[1]

    #     first_prop = properties[i]
    #     second_prop = properties[nn_label-1]

    #     # skip intersect if distance is zero
    #     if(nn_distance == 0.0):
    #         try:
    #             properties_wo_overlap.remove(first_prop)
    #             properties_wo_overlap.remove(second_prop)
    #         except ValueError:
    #              pass  # do nothing
    #         continue
    #     if(intersects(first_prop.bbox, second_prop.bbox)):
    #         try:
    #             properties_wo_overlap.remove(first_prop)
    #             properties_wo_overlap.remove(second_prop)
    #         except ValueError:
    #              pass  # do nothing

    # print(f'Properties remaining = {len(properties_wo_overlap)}')


    # Compute metric for best colonies
    # Defined as:
    #     max distance from nn * area * (1 - eccentricity)
    quality_metrics = []
    quality_metrics = np.empty(len(properties_wo_overlap), dtype=object)
    for i in range(0, len(properties_wo_overlap)):
        # try:
        #     nn = min_distance_nn[p.label - 1]
        # except IndexError:
        #     continue
        nn_distance = min_distance_nn[i]
        p = properties_wo_overlap[i]
        quality_metrics[i] = (p.area * nn_distance * (1.0 - p.eccentricity), p.label)
        if (p.area <= 100):
            print(f'label {p.label} to small <= 200')
            quality_metrics[i] = (0, p.label)
    # for i in range(0, len(properties_wo_overlap)):
    #     min_distance = min_distance_nn[i]
    #     p = properties_wo_overlap[i]
    #     quality_metrics.append((p.area * min_distance * (1.0 - p.eccentricity), p.label))




    # reverse_metrics = quality_metrics[::-1]
    reverse_metrics = sorted(quality_metrics, key=lambda x: (-x[0], x[1]))
    # quality_metrics.sort(reverse = True)

    print("plotting")

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

        ax[8].imshow(img_original_cropped)
        ax[8].set_title(f'Cell\'s Centroid N={len(properties)}')
        # add X marks on centroids
        for p in properties:
            ax[8].scatter(p.centroid[1], p.centroid[0], s=20, c='red', marker='x', linewidth=1)

        ax[9].imshow(img_original_cropped)
        ax[9].set_title(f'Overlapping cell\'s removed N={len(properties_wo_overlap)}')
        # add X marks on centroids
        for p in properties_wo_overlap:
            ax[9].scatter(p.centroid[1], p.centroid[0], s=20, c='red', marker='x', linewidth=1)

        ax[10].imshow(img_original_cropped)
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

        plt.tight_layout()
        plt.show()


    # Plot region properties interactively
    if (plot_interactive_properties):
        property_names = ['area', 'eccentricity', 'perimeter']

        print("Generating region properties interactively plot, this may take some time...")
        # plot_region_roperties(img, img_labels, properties_wo_overlap, property_names)
        plot_region_roperties(img_original_cropped, img_labels, properties, property_names)
