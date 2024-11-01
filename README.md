# Colony Morphology
Python3 library to analyse culture characteristics of a bacterial colony on an agar plate.

Apply a mask based on the petri dish radius and use the watershed algorithm to find the cell's. Finally, annotate the best colonies based on cell's metrics: area, diameter, eccentricity, min distance from near neighbor's.

<p align="center">
  <img src="./result/oneplus_hd1900_1_annotated.png" width="50%" alt="Annotation Result from OnePlus HD1900 Camera"/>
</p>

| ID | Area | Diameter | Eccentricity | Min NN Distance | Centroid         | Label |
|----|------|----------|--------------|-----------------|------------------|-------|
| 1  | 451  | 23.96    | 0.49         | 119.77          | 258.62, 736.92   | 1020  |
| 2  | 544  | 26.31    | 0.41         | 110.53          | 963.69, 428.5    | 2383  |
| 3  | 489  | 24.95    | 0.37         | 108.63          | 329.33, 1428.18  | 1179  |
| 4  | 518  | 25.68    | 0.25         | 88.28           | 475.36, 884.16   | 1533  |
| 5  | 460  | 24.2     | 0.42         | 90.56           | 361.56, 1102.45  | 1265  |
| 6  | 578  | 27.12    | 0.33         | 58.12           | 457.72, 1630.32  | 1490  |
| 7  | 395  | 22.42    | 0.36         | 89.90           | 1639.70, 1242.21 | 4310  |
| 8  | 487  | 24.9     | 0.47         | 83.71           | 689.23, 315.88   | 1932  |
| 9  | 545  | 26.34    | 0.42         | 71.15           | 223.18, 1134.33  | 905   |
| 11 | 482  | 24.77    | 0.82         | 56.02           | 826.56, 423.97   | 2192  |
| 12 | 505  | 25.35    | 0.31         | 59.18           | 780.18, 1789.22  | 2134  |


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

Try playing with different [parameters](./demo/demo.py#L30-L35), you can also enable different [plots](./demo/demo.py#L51-L55):

``` sh
# realsense camera
$ python3 demo/demo.py --image_path=dataset/intel_rs_d415_1.png --dish_radius=400 --cell_min_radius=3 --cell_max_radius=10

$ python3 demo/demo.py --image_path=dataset/intel_rs_d415_2.png --dish_radius=400 --cell_min_radius=3 --cell_max_radius=10

# oneplus camera
$ python3 demo/demo.py --image_path=dataset/oneplus_hd1900_1.jpg --dish_radius=1050 --dish_inner_offset=200 --cell_min_radius=6 --cell_max_radius=26

```

## Results
Check the [result](./result) folder for annotated and cropped images from the demo.

