import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyefd
import os

from glob import glob
from sklearn.decomposition import PCA


def get_largest_contour(segmented_image: np.ndarray):
    """

    :param segmented_image:
    :return:
    """

    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return np.squeeze(largest_contour)


def elliptic_fourier_w_norm(segmented_image: np.ndarray):
    """

    :param segmented_image:
    :return:
    """

    largest_contour = get_largest_contour(segmented_image)
    coefficients = pyefd.elliptic_fourier_descriptors(largest_contour, order=15, normalize=True)
    np.testing.assert_almost_equal(coefficients[0, 0], 1.0, decimal=14)
    np.testing.assert_almost_equal(coefficients[0, 1], 0.0, decimal=14)
    np.testing.assert_almost_equal(coefficients[0, 2], 0.0, decimal=14)
    return coefficients.flatten()[3:]


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses an input image by applying edge detection, thresholding, dilation, and connected components
    analysis.

    :param image: Input image.
    :return: Preprocessed image.
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = cv2.add(np.abs(sobel_x), np.abs(sobel_y))

    _, thresholded_edge_map = cv2.threshold(edge_map, 10, 255, cv2.THRESH_BINARY)
    dilated_edge_map = cv2.dilate(thresholded_edge_map, kernel=np.ones((5, 5), np.uint8), iterations=1)
    dilated_edge_map = dilated_edge_map.astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(dilated_edge_map)

    sorted_indices = np.argsort(stats[:, 4])[::-1]
    second_largest_index = sorted_indices[1]

    pill_mask = np.zeros_like(dilated_edge_map)
    pill_mask[labels == second_largest_index] = 255

    return cv2.bitwise_and(image, image, mask=pill_mask)


def plot_vectors(class_averages):
    """

    :param class_averages:
    :return:
    """

    classes = list(class_averages.keys())
    vectors = list(class_averages.values())

    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2],
               c=range(len(classes)),
               s=80,
               marker='o',
               cmap="viridis",
               alpha=0.7)
    ax.set_zlim3d(-1, 1)

    for i, label in enumerate(classes):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2] + 0.02,
                label, ha='center', va='center')

    plt.title('Point Cloud After Dimensionality Reduction')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()


def main():
    images_path = sorted(glob("pills/*.JPG"))

    pill_dict = {}

    for image_path in images_path:
        new_image_original = cv2.imread(image_path, 1)
        image_name = os.path.basename(image_path)
        image_name = image_name.split(".")[0]
        new_image = cv2.cvtColor(new_image_original, cv2.COLOR_BGR2GRAY)
        new_segmented_image = preprocess_image(new_image)
        coeffs = elliptic_fourier_w_norm(new_segmented_image)
        pill_dict[image_name] = coeffs

    plot_vectors(pill_dict)


if __name__ == "__main__":
    main()
