# Colony Morphology
Python3 library to analyse culture characteristics of a bacterial colony on an agar plate.

Apply a mask based on the petri dish radius and use the watershed algorithm to find the cell's. Finally, annotate the best colonies based on cell's metrics: area, diameter, eccentricity, min distance from near neighbor's.

<p align="center">
  <img src="./result/oneplus_hd1900_1_annotated.png" width="50%" alt="Annotation Result from OnePlus HD1900 Camera"/>
</p>

| ID | Label | Cell Quality | Area | Diameter | Eccentricity | Compactness | Solidity | Min NN Distance | Centroid (x) | Centroid (y) |
|----|-------|--------------|------|----------|--------------|-------------|----------|-----------------|--------------|--------------|
| 1  | 19    | 41413        | 501  | 25.25    | 0.48         | 0.87        | 0.95     | 159             | 736.97       | 329.64       |
| 2  | 368   | 40484        | 600  | 27.63    | 0.38         | 0.88        | 0.95     | 110             | 428.41       | 1034.63      |
| 3  | 23    | 39227        | 542  | 26.26    | 0.35         | 0.86        | 0.95     | 112             | 1428.20      | 400.34       |
| 4  | 74    | 36612        | 563  | 26.77    | 0.26         | 0.92        | 0.96     | 88              | 884.20       | 546.37       |
| 5  | 30    | 27302        | 508  | 25.43    | 0.40         | 0.90        | 0.96     | 90              | 1102.44      | 432.50       |
| 6  | 67    | 26484        | 639  | 28.52    | 0.30         | 0.92        | 0.96     | 59              | 1630.33      | 528.76       |
| 7  | 1125  | 24751        | 446  | 23.82    | 0.37         | 0.91        | 0.95     | 88              | 1242.28      | 1710.71      |
| 8  | 176   | 22472        | 540  | 26.22    | 0.44         | 0.90        | 0.95     | 74              | 315.84       | 760.21       |
| 9  | 15    | 19805        | 608  | 27.82    | 0.43         | 0.85        | 0.94     | 58              | 1134.39      | 294.38       |
| 10 | 57    | 19145        | 427  | 23.31    | 0.07         | 0.92        | 0.95     | 48              | 532.59       | 489.83       |

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
$ python3 demo/demo.py --image_path=dataset/intel_rs_d415_1.png --dish_radius=441 --dish_inner_offset=67

$ python3 demo/demo.py --image_path=dataset/intel_rs_d415_2.png --dish_radius=441 --dish_inner_offset=67

# oneplus camera
$ python3 demo/demo.py --image_path=dataset/oneplus_hd1900_1.jpg --dish_radius=1050 --dish_inner_offset=200 --cell_min_radius=6

```

## Results
Check the [result](./result) folder for annotated and cropped images from the demo.

