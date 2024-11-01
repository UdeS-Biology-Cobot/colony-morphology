# Colony Morphology
Python3 library to analyse culture characteristics of a bacterial colony on an agar plate.

Apply a mask based on the petri dish radius and use the watershed algorithm to find the cell's. Finally, annotate the best colonies based on cell's metrics: area, diameter, eccentricity, min distance from near neighbor's.

<p align="center">
  <img src="./result/oneplus_hd1900_1_annotated.png" width="50%" alt="Annotation Result from OnePlus HD1900 Camera"/>
</p>

| ID | Area | Diameter | Eccentricity | Min NN Distance | Centroid         | Label |
|----|------|----------|--------------|-----------------|------------------|-------|
| 1  | 544  | 26.31    | 0.41         | 110.53          | 963.69, 428.5    | 2383  |
| 2  | 518  | 25.68    | 0.25         | 88.28           | 475.36, 884.16   | 1533  |
| 3  | 489  | 24.95    | 0.37         | 108.63          | 329.33, 1428.18  | 1179  |
| 4  | 563  | 26.77    | 0.31         | 73.27           | 1109.12, 224.09  | 2607  |
| 5  | 451  | 23.96    | 0.49         | 119.77          | 258.62, 736.92   | 1020  |
| 6  | 460  | 24.2     | 0.42         | 90.56           | 361.56, 1102.45  | 1265  |
| 7  | 395  | 22.42    | 0.36         | 89.90           | 1639.70, 1242.21 | 4310  |
| 8  | 545  | 26.34    | 0.42         | 71.15           | 223.18, 1134.33  | 905   |
| 9  | 578  | 27.12    | 0.33         | 58.12           | 457.72, 1630.32  | 1490  |
| 10 | 519  | 25.7     | 0.29         | 58.92           | 396.84, 655.35   | 1366  |
| 11 | 487  | 24.9     | 0.47         | 83.71           | 689.23, 315.88   | 1932  |
| 12 | 505  | 25.35    | 0.31         | 59.18           | 780.18, 1789.22  | 2134  |
| 13 | 439  | 23.64    | 0.29         | 59.23           | 307.50, 1175.10  | 1141  |
| 14 | 401  | 22.59    | 0.28         | 58.41           | 637.58, 1547.02  | 1829  |
| 15 | 482  | 24.77    | 0.82         | 56.02           | 826.56, 423.97   | 2192  |


## Dependencies
Install dependencies:
``` sh
$ pip install -r requirements
```

## Build
Build library:
``` sh
$ pip install .
```

## Run

Try different parameters values, you can also plot different sequences:
- Petri dish mask
- Segmentation sequence

``` sh
# realsense camera
$ python3 demo/demo.py --image_path=dataset/intel_rs_d415_1.png --dish_radius=400 --cell_min_radius=3 --cell_max_radius=10

$ python3 demo/demo.py --image_path=dataset/intel_rs_d415_2.png --dish_radius=400 --cell_min_radius=3 --cell_max_radius=10

# oneplus camera
$ python3 demo/demo.py --image_path=dataset/oneplus_hd1900_1.jpg --dish_radius=1000 --cell_min_radius=6

```

## Results
Check the [result](./result) folder for annotated and cropped images from the demo.

