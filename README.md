# Colony Morphology
Python3 library to analyse culture characteristics of a bacterial colony on an agar plate.

Apply a mask based on the petri dish radius and use the watershed algorithm to find the cell's. Finally, annotate the best colonies based on cell's metrics: area, diameter, eccentricity, min distance from near neighbor's.

<p align="center">
  <img src="./result/oneplus_hd1900_1/annotated_cell.png" width="50%" alt="Annotation Result from OnePlus HD1900 Camera"/>
</p>

| ID | Label | Cell Quality | Area | Diameter | Eccentricity | Compactness | Solidity | Collision Distance | Centroid (x) | Centroid (y) |
|----|-------|--------------|------|----------|--------------|-------------|----------|--------------------|--------------|--------------|
| 1  | 562   | 0.78         | 701  | 30       | 0.32         | 0.89        | 0.95     | 69                 | 842          | 289          |
| 2  | 178   | 0.75         | 666  | 29       | 0.27         | 0.91        | 0.96     | 60                 | 353          | 745          |
| 3  | 27    | 0.75         | 651  | 29       | 0.30         | 0.91        | 0.95     | 60                 | 207          | 1289         |
| 4  | 1376  | 0.71         | 526  | 26       | 0.31         | 0.86        | 0.94     | 65                 | 1517         | 1103         |
| 5  | 277   | 0.70         | 682  | 29       | 0.48         | 0.79        | 0.93     | 57                 | 474          | 186          |
| 6  | 332   | 0.69         | 596  | 27       | 0.53         | 0.73        | 0.93     | 65                 | 567          | 177          |
| 7  | 164   | 0.68         | 728  | 30       | 0.30         | 0.86        | 0.95     | 32                 | 335          | 1491         |
| 8  | 16    | 0.67         | 688  | 30       | 0.40         | 0.78        | 0.93     | 43                 | 101          | 995          |
| 9  | 457   | 0.64         | 608  | 28       | 0.32         | 0.73        | 0.93     | 42                 | 704          | 284          |
| 10 | 25    | 0.64         | 560  | 27       | 0.34         | 0.89        | 0.95     | 33                 | 186          | 1036         |

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

Try playing with different [parameters](./demo/demo.py#L11-L44), you can also enable different [output options](./demo/demo.py#L48-L59):

``` sh
# realsense camera
$ python3 demo/demo.py --image=dataset/intel_rs_d415_1.png --dish-diameter=882 --dish-offset=67

$ python3 demo/demo.py --image=dataset/intel_rs_d415_2.png --dish-diameter=882 --dish-offset=67

# oneplus camera
$ python3 demo/demo.py --image=dataset/oneplus_hd1900_1.jpg --dish-diameter=2100 --dish-offset=200 --cell-min-diameter=12
```

## Results
Check the [result](./result) folder for annotated and cropped images from the demo.

