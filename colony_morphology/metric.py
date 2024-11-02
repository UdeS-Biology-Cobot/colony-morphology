import math

# http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
def compactness(area, perimeter):
    """
    Ratio of the area of an object to the area of a
    circle with the same perimeter.

    - A circle is used as it is the object with the most
      compact shape
    - The measure takes a maximum value of 1 for a circle
    - A square has compactness = PI/4
    - Objects which have an elliptical shape, or a
      boundary that is irregular rather than smooth, will
      decrease the measure.
    - Objects that have complicated, irregular
      boundaries have larger compactness.

    Parameters
    -----------
    area: float
    perimeter: float

    Returns
    --------
    float
        The compactness metric.
    """
    return 4*math.pi*area / (perimeter*perimeter)
