from matplotlib import patches
from matplotlib import pyplot as plt

import plotly
import plotly.express as px
import plotly.graph_objects as go

from skimage import measure

def plot_bboxes(image, bboxes, xywh: bool = True, labels = None) -> None:
    """
    Args:
      image_file: str specifying the image file path
      bboxes: list of bounding box annotations for all the detections
      xywh: bool, if True, the bounding box annotations are specified as
        [xmin, ymin, width, height]. If False the annotations are specified as
        [xmin, ymin, xmax, ymax]. If you are unsure what the mode is try both
        and check the saved image to see which setting gives the
        correct visualization.

    """
    fig = plt.figure()

    # add axes to the image
    ax = fig.add_axes([0, 0, 1, 1])

    plt.imshow(image)

    # Iterate over all the bounding boxes
    for i, bbox in enumerate(bboxes):
        if xywh:
          xmin, ymin, w, h = bbox.bbox
        else:
          ymin, xmin, ymax, xmax = bbox.bbox
          w = xmax - xmin
          h = ymax - ymin

        # add bounding boxes to the image
        box = patches.Rectangle(
            (xmin, ymin), w, h, edgecolor="red", facecolor="none"
        )

        ax.add_patch(box)

        if labels is not None:
          rx, ry = box.get_xy()
          cx = rx + box.get_width()/2.0
          cy = ry + box.get_height()/8.0
          l = ax.annotate(
            labels[i],
            (cx, cy),
            fontsize=8,
            fontweight="bold",
            color="white",
            ha='center',
            va='center'
          )
          l.set_bbox(
            dict(facecolor='red', alpha=0.5, edgecolor='red')
          )

    plt.axis('off')
    # outfile = os.path.join(image_folder, "image_bbox.png")
    # fig.savefig(outfile)

    # print("Saved image with detections to %s" % outfile)

    plt.show()




def plot_region_roperties(image, labels, properties, property_names):
    """
    Plot selected properties interactively.
    Uses plotly in order to display properties when hovering over the objects.
    """
    fig = px.imshow(image, binary_string=True)
    fig.update_traces(hoverinfo='skip')  # hover is only for label info

    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    for index in range(1, len(properties)):
        label_i = properties[index].label
        contour = measure.find_contours(labels == label_i, 0.5)[0]
        y, x = contour.T
        hoverinfo = ''
        for prop_name in property_names:
            hoverinfo += f'<b>{prop_name}: {getattr(properties[index], prop_name):.2f}</b><br>'
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=label_i,
                mode='lines',
                fill='toself',
                showlegend=False,
                hovertemplate=hoverinfo,
                hoveron='points+fills',
            )
        )

    plotly.io.show(fig)
