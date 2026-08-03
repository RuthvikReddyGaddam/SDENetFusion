import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import filters
from skimage.graph import rag_boundary
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries, slic


def superpixel_segmentation(
    image,
    n_segments=1000,
    compactness=10,
):
    return slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1,
        enforce_connectivity=True,
        convert2lab=True,
        start_label=0,
        channel_axis=-1,
    )


def draw_superpixel_graph(
    image,
    segments,
    rag,
    show_boundaries=True,
    edge_width=1.5,
    node_size=12,
):
    """
    Draws RAG edges between superpixel centroids.

    Args:
        image:
            RGB image with shape [H, W, 3].

        segments:
            Superpixel label map with shape [H, W].

        rag:
            Graph produced by skimage.graph.rag_boundary().

        show_boundaries:
            Show green superpixel boundaries.

        edge_width:
            Width of graph edges.

        node_size:
            Centroid marker size.
    """

    if show_boundaries:
        display_image = mark_boundaries(
            image,
            segments,
            color=(0, 1, 0),
            mode="thick",
        )
    else:
        display_image = image

    # regionprops returns centroid as (row, column), equivalent to (y, x).
    centroids = {
        region.label: region.centroid
        for region in regionprops(segments + 1)
    }

    # regionprops labels segments + 1, so convert keys back to 0-based IDs.
    centroids = {
        label - 1: centroid
        for label, centroid in centroids.items()
    }

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.imshow(display_image)

    # Draw graph edges.
    for source_node, target_node in rag.edges():
        source_y, source_x = centroids[source_node]
        target_y, target_x = centroids[target_node]

        ax.plot(
            [source_x, target_x],
            [source_y, target_y],
            linewidth=edge_width,
            alpha=0.65,
        )

    # Draw graph nodes at superpixel centroids.
    node_x = []
    node_y = []

    for node_id in rag.nodes():
        centroid_y, centroid_x = centroids[node_id]
        node_x.append(centroid_x)
        node_y.append(centroid_y)

    ax.scatter(
        node_x,
        node_y,
        s=node_size,
        zorder=3,
    )

    ax.set_title(
        f"Superpixel region-adjacency graph\n"
        f"Nodes: {rag.number_of_nodes()} | "
        f"Edges: {rag.number_of_edges()}"
    )
    ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = (
        r"C:\Users\gadda\Documents\FeatureFusionMoNuSAC"
        r"\SDENet-Fusion\Datasets\Glas\train\img\train_9.bmp"
    )

    # OpenCV loads BGR.
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # SLIC works well with float images in [0, 1].
    image_float = image_rgb.astype(np.float32) / 255.0

    segments = superpixel_segmentation(
        image_float,
        n_segments=1000,
        compactness=10,
    )

    gray_image = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32) / 255.0

    edge_map = filters.sobel(gray_image)

    rag = rag_boundary(
        segments,
        edge_map,
        connectivity=1,
    )

    print("Generated segments:", len(np.unique(segments)))
    print("RAG nodes:", rag.number_of_nodes())
    print("RAG edges:", rag.number_of_edges())
    print("First 20 edges:", list(rag.edges())[:20])

    draw_superpixel_graph(
        image=image_float,
        segments=segments,
        rag=rag,
        show_boundaries=True,
        edge_width=1.5,
        node_size=15,
    )