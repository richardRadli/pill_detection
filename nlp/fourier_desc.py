import cv2
import numpy as np
import pyefd

from glob import glob
from scipy.spatial.distance import directed_hausdorff


def elliptic_fourier(image, index):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour = np.squeeze(largest_contour)

    number_of_points = largest_contour.shape[0]
    locus = pyefd.calculate_dc_coefficients(largest_contour)
    coefficients = pyefd.elliptic_fourier_descriptors(largest_contour, order=5)

    rotated_image = cv2.transpose(image)
    pyefd.plot_efd(index, coefficients, locus, rotated_image, largest_contour)

    reconstruction = pyefd.reconstruct_contour(coefficients, locus, number_of_points)
    hausdorff_distance, _, _ = directed_hausdorff(reconstruction, largest_contour)
    print(hausdorff_distance)


def main():
    image_path = "C:/Users/ricsi/Documents/project/storage/IVM/datasets/ogyei_v2/single/splitted/test/gt_test_masks"
    images = sorted(glob(image_path + "/*.png"))

    for i, img_path in enumerate(images):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        elliptic_fourier(image, i)


if __name__ == "__main__":
    main()
