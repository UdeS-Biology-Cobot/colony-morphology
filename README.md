# Colony Morphology
Python3 library to analyse culture characteristics of a bacterial colony on an agar plate.

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
