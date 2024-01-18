import cv2
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import pyefd
import os

from scipy.spatial.distance import directed_hausdorff, euclidean, pdist, squareform


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class FourierDescriptor:
    def __init__(self, load, order):
        self.order = order
        self.load = load
        self.file_path = "C:/Users/ricsi/Desktop/Fourier_desc/collected_images_by_shape_nih"
        self.json_filename = f"C:/Users/ricsi/Desktop/Fourier_desc/average_{self.order}.json"
        self.query_image_path = \
            ("C:/Users/ricsi/Documents/project/storage/IVM/datasets/nih/ref/"
             "00002322930/W9WJ5DVPKN8-DARNMB53SKVF8PU!2N.JPG")

    @staticmethod
    def plot_efd(filename, coeffs, locus=(0.0, 0.0), image=None, contour=None, n=300):
        N = coeffs.shape[0]
        N_half = int(np.ceil(N / 2))
        n_rows = 2

        t = np.linspace(0, 1.0, n)
        xt = np.ones((n,)) * locus[0]
        yt = np.ones((n,)) * locus[1]

        for n in range(coeffs.shape[0]):
            xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + (
                    coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t)
            )
            yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + (
                    coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t)
            )
            ax = plt.subplot2grid((n_rows, N_half), (n // N_half, n % N_half))
            ax.set_title(str(n + 1))

            if image is not None:
                # A background image of shape [rows, cols] gets transposed by imshow so that the first dimension is
                # vertical and the second dimension is horizontal. This implies swapping the x and y axes when plotting
                # a curve.
                if contour is not None:
                    ax.plot(contour[:, 1], contour[:, 0], "c--", linewidth=2)
                ax.plot(yt, xt, "r", linewidth=2)
                ax.imshow(image, cmap="gray")
            else:
                # Without a background image, no transpose is implied. This case is useful when (x,y) point clouds
                # without relation to an image are to be handled.
                if contour is not None:
                    ax.plot(contour[:, 0], contour[:, 1], "c--", linewidth=2)
                ax.plot(xt, yt, "r", linewidth=2)
                ax.axis("equal")

        # plt.show()
        plt.tight_layout()
        plt.savefig(f"C:/Users/ricsi/Desktop/Fourier_desc/fourier/{filename}")
        plt.close()
        gc.collect()

    @staticmethod
    def preprocess_image(image):
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edge_map = cv2.add(np.abs(sobel_x), np.abs(sobel_y))

        _, thresholded_edge_map = cv2.threshold(edge_map, 10, 255, cv2.THRESH_BINARY)
        dilated_edge_map = cv2.dilate(thresholded_edge_map, kernel=np.ones((5, 5), np.uint8), iterations=1)
        dilated_edge_map = dilated_edge_map.astype(np.uint8)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_edge_map)

        sorted_indices = np.argsort(stats[:, 4])[::-1]
        second_largest_index = sorted_indices[1]

        pill_mask = np.zeros_like(dilated_edge_map)
        pill_mask[labels == second_largest_index] = 255

        return cv2.bitwise_and(image, image, mask=pill_mask)

    def elliptic_fourier(self, image, segmented_image, normalize, filename=None):
        contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour = np.squeeze(largest_contour)

        if normalize:
            coefficients = pyefd.elliptic_fourier_descriptors(largest_contour, order=self.order, normalize=True)
            np.testing.assert_almost_equal(coefficients[0, 0], 1.0, decimal=14)
            np.testing.assert_almost_equal(coefficients[0, 1], 0.0, decimal=14)
            np.testing.assert_almost_equal(coefficients[0, 2], 0.0, decimal=14)
            return coefficients.flatten()[3:]
        else:
            number_of_points = largest_contour.shape[0]
            locus = pyefd.calculate_dc_coefficients(largest_contour)
            coefficients = pyefd.elliptic_fourier_descriptors(largest_contour, order=self.order, normalize=False)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rotated_image = cv2.transpose(image)
            self.plot_efd(filename, coefficients, locus, rotated_image, largest_contour)

            reconstruction = pyefd.reconstruct_contour(coefficients, locus, number_of_points)
            hausdorff_distance, _, _ = directed_hausdorff(reconstruction, largest_contour)
            print(hausdorff_distance)

    def plot_euclidiean_distances(self, class_averages):
        class_labels = list(class_averages.keys())
        class_vectors = list(class_averages.values())

        distances = squareform(pdist(class_vectors, 'euclidean'))

        plt.figure(figsize=(8, 8))
        plt.imshow(distances, cmap='viridis', vmin=0, vmax=np.max(distances))
        plt.colorbar(label='Euclidean Distance')

        plt.xticks(np.arange(len(class_labels)), class_labels, rotation=45, ha='right')
        plt.yticks(np.arange(len(class_labels)), class_labels)
        plt.xlabel('Class Label')
        plt.ylabel('Class Label')
        plt.title('Pairwise Euclidean Distances between Class Averages')

        # Show the plot
        plt.tight_layout()
        plt.savefig(f"C:/Users/ricsi/Desktop/Fourier_desc/euclidean_distances_{self.order}.png", dpi=300)
        plt.close()

    @staticmethod
    def calculate_distance(vector1, vector2):
        return euclidean(vector1, vector2)

    def compare_distances(self, class_averages):
        new_image_original = cv2.imread(self.query_image_path, 1)
        new_image = cv2.cvtColor(new_image_original, cv2.COLOR_BGR2GRAY)
        new_segmented_image = self.preprocess_image(new_image)
        new_fourier_descriptor = self.elliptic_fourier(image=new_image_original,
                                                       segmented_image=new_segmented_image,
                                                       normalize=True)

        closest_class = None
        min_distance = float('inf')

        for class_label, class_vector in class_averages.items():
            distance = self.calculate_distance(new_fourier_descriptor, class_vector)
            print(f"Distance to {class_label}: {distance}")

            if distance < min_distance:
                min_distance = distance
                closest_class = class_label

        print(f"\nThe closest class to the query image is: {closest_class}")

    def main(self):
        if not self.load:
            class_averages = {}

            for root, dirs, files in os.walk(self.file_path):
                for class_label in dirs:
                    class_coefficients = []

                    class_path = os.path.join(root, class_label)
                    for file in os.listdir(class_path):
                        image_path = os.path.join(class_path, file)
                        image_original = cv2.imread(image_path, 1)
                        try:
                            image = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
                            segmented_image = self.preprocess_image(image)
                            norm_fourier_coeff = self.elliptic_fourier(image=image_original,
                                                                       segmented_image=segmented_image,
                                                                       normalize=True,
                                                                       filename=file)
                            class_coefficients.append(norm_fourier_coeff)
                        except Exception as e:
                            print(f"{e}, Could not open{image_path}")

                    if class_coefficients:
                        class_average = np.mean(class_coefficients, axis=0)
                        class_averages[class_label] = class_average

            self.plot_euclidiean_distances(class_averages)

            with open(self.json_filename, "w") as json_file:
                json.dump(class_averages, json_file, cls=NumpyEncoder)
        else:
            try:
                with open(self.json_filename, "r") as json_file:
                    class_averages = json.load(json_file)
                self.compare_distances(class_averages)
            except FileNotFoundError as fne:
                print(f"{fne}")


if __name__ == "__main__":
    try:
        fd = FourierDescriptor(load=True, order=5)
        fd.main()
    except KeyboardInterrupt:
        print("Ctrl+C pressed")
