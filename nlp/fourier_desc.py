import colorama
import cv2
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import shutil
import pyefd
import os
import openpyxl

from datetime import datetime
from glob import glob
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff, euclidean, pdist, squareform


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class FourierDescriptor:
    def __init__(self, copy_images, load, order):
        self.order = order
        self.copy_images = copy_images
        self.load = load

        self.images_dir = "C:/Users/ricsi/Documents/project/storage/IVM/datasets/nih/ref"

        self.file_path = "C:/Users/ricsi/Desktop/Fourier_desc/collected_images_by_shape_nih"
        os.makedirs(self.file_path, exist_ok=True)

        self.plot_efd_dir = "C:/Users/ricsi/Desktop/Fourier_desc/fourier/"
        os.makedirs(self.plot_efd_dir, exist_ok=True)

        self.plot_euc_dir = "C:/Users/ricsi/Desktop/Fourier_desc/euclidean_distance/"
        os.makedirs(self.plot_euc_dir, exist_ok=True)

        self.json_filename = f"C:/Users/ricsi/Desktop/Fourier_desc/average_{self.order}.json"
        self.query_image_path = \
            ("C:/Users/ricsi/Documents/project/storage/IVM/datasets/nih/ref/"
             "00007365022_1\\ZPUR13YZJ0KVFXFFI06VPSG7T3VJ4L.JPG")
        self.excel_path = "C:/Users/ricsi/Documents/project/storage/IVM/datasets/nih/xlsxl/ref.xlsx"

        self.timestamp = self.create_timestamp()
        colorama.init()

    @staticmethod
    def create_timestamp() -> str:
        """
        Creates a timestamp in the format of '%Y-%m-%d_%H-%M-%S', representing the current date and time.

        :return: The timestamp string.
        """

        return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def get_excel_values(self) -> None:
        """
        Reads an Excel file, processes the data, and copies corresponding files based on specified conditions.

        :return: None
        """

        workbook = openpyxl.load_workbook(self.excel_path)
        sheet = workbook['Sheet1']

        shape_dict = {}

        for row in sheet.iter_rows(min_row=2, values_only=True):
            ndc11 = row[0]
            shape = row[5]

            if shape in shape_dict:
                shape_dict[shape].append(ndc11)
            else:
                shape_dict[shape] = [ndc11]

        workbook.close()

        for shape, ndc11_list in tqdm(shape_dict.items(), desc=colorama.Fore.GREEN + "Processing Shapes"):
            print(f"Shape: {shape}")
            for ndc11 in ndc11_list:
                ndc11 = self.extend_ndc11_number(ndc11)
                src_dir = os.path.join(self.images_dir, str(ndc11))
                src_file = None
                try:
                    files = os.listdir(src_dir)[0]
                    src_file = os.path.join(src_dir, files)
                    dst_dir = os.path.join(self.file_path, shape)
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy(src_file, dst_dir)
                except FileNotFoundError:
                    print(f"File not found: {src_file}")

    def plot_efd(self, filename: str, coeffs: np.ndarray, locus: tuple = (0.0, 0.0), image: np.ndarray = None,
                 contour: np.ndarray = None, n: int = 300) -> None:
        """
         Plots the elliptical Fourier descriptors.

         :param filename: Name of the file to save the plot.
         :param coeffs: Coefficients for elliptical Fourier descriptors.
         :param locus: Tuple representing the locus (center) of the plot.
         :param image: Background image (optional).
         :param contour: Contour data (optional).
         :param n: Number of points to generate the plot.
         :return: None
         """

        num_of_coeffs = coeffs.shape[0]
        n_half = int(np.ceil(num_of_coeffs / 2))
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
            ax = plt.subplot2grid((n_rows, n_half), (n // n_half, n % n_half))
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
        plt.savefig(os.path.join(self.plot_efd_dir, f"{filename}"))
        plt.close()
        gc.collect()

    @staticmethod
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
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_edge_map)

        sorted_indices = np.argsort(stats[:, 4])[::-1]
        second_largest_index = sorted_indices[1]

        pill_mask = np.zeros_like(dilated_edge_map)
        pill_mask[labels == second_largest_index] = 255

        return cv2.bitwise_and(image, image, mask=pill_mask)

    def elliptic_fourier(self, image: np.ndarray, segmented_image: np.ndarray, normalize: bool, filename: str = None):
        """
        Compute elliptic Fourier descriptors for the largest contour in the segmented image.

        :param image: Original image.
        :param segmented_image: Segmented image.
        :param normalize: Flag indicating whether to normalize coefficients.
        :param filename: Name of the file for plot (optional).
        :return: Optional
        """

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

    def plot_euclidean_distances(self, class_averages: dict) -> None:
        """
        Plot pairwise Euclidean distances between class averages.

        :param class_averages: Dictionary mapping class labels to class averages.
        :return: None
        """

        class_labels = list(class_averages.keys())
        class_vectors = np.array(list(class_averages.values()))

        normalized_values = ((class_vectors - np.min(class_vectors, axis=0)) /
                             (np.max(class_vectors, axis=0) - np.min(class_vectors, axis=0)))

        distances = squareform(pdist(normalized_values, 'euclidean'))

        plt.figure(figsize=(8, 8))
        plt.imshow(distances, cmap='viridis', vmin=0, vmax=np.max(distances))
        plt.colorbar(label='Euclidean Distance')

        plt.xticks(np.arange(len(class_labels)), class_labels, rotation=45, ha='right')
        plt.yticks(np.arange(len(class_labels)), class_labels)
        plt.xlabel('Class Label')
        plt.ylabel('Class Label')
        plt.title('Pairwise Euclidean Distances between Class Averages')

        plt.tight_layout()
        filename = os.path.join(self.plot_euc_dir, f"euclidean_distances_{self.order}_{self.timestamp}.png")
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def calculate_distance(vector1: list, vector2: list) -> float:
        """
        Calculate the Euclidean distance between two vectors.

        :param vector1: First vector.
        :param vector2: Second vector.
        :return: Euclidean distance between the two vectors.
        """

        return euclidean(vector1, vector2)

    @staticmethod
    def extend_ndc11_number(ndc11_id):
        """
        Extend an NDC11 number by adding leading zeros if needed.

        :param ndc11_id: NDC11 number to extend.
        :return: Extended NDC11 number as a string.
        """

        ndc11_id = str(ndc11_id)

        if ndc11_id.endswith('_1'):
            ndc11_id = int(ndc11_id[:-2])
        elif ndc11_id.endswith('_2'):
            ndc11_id = int(ndc11_id[:-2])
        else:
            ndc11_id = ndc11_id

        if len(str(ndc11_id)) != 11:
            num_zeros = 11 - len(str(ndc11_id))
            ndc11_id = "0" * num_zeros + str(ndc11_id)

        return ndc11_id

    def compare_distances(self, class_averages, image_path):
        """
        Compare the Fourier descriptor of a new image with class averages and print the closest match.

        :param class_averages: Dictionary mapping class labels to class averages.
        :return: None
        """
        if image_path.endswith('.JPG'):
            hit = 0
            miss_hit = 0
            new_image_original = cv2.imread(image_path, 1)
            new_image = cv2.cvtColor(new_image_original, cv2.COLOR_BGR2GRAY)
            new_segmented_image = self.preprocess_image(new_image)
            new_fourier_descriptor = self.elliptic_fourier(image=new_image_original,
                                                           segmented_image=new_segmented_image,
                                                           normalize=True)

            closest_class = None
            min_distance = float('inf')

            for class_label, class_vector in class_averages.items():
                distance = self.calculate_distance(new_fourier_descriptor, class_vector)
                # print(f"Distance to {class_label}: {distance}")

                if distance < min_distance:
                    min_distance = distance
                    closest_class = class_label

            workbook = openpyxl.load_workbook(self.excel_path)
            sheet = workbook['Sheet1']
            original_shape = []

            query_ndc11 = image_path.split("\\")[0].split("/")[-1]
            query_ndc11 = self.extend_ndc11_number(query_ndc11)

            for row in sheet.iter_rows(min_row=2, values_only=True):
                ndc11 = row[0]
                shape = row[5]

                ndc11 = self.extend_ndc11_number(ndc11)
                if query_ndc11 == ndc11:
                    original_shape.append(shape)

            if original_shape[0] == closest_class:
                print(colorama.Fore.LIGHTGREEN_EX +
                      f"The predicted shape {closest_class} matches the original shape {original_shape[0]}")
            else:
                print(colorama.Fore.LIGHTRED_EX +
                      f"The predicted shape {closest_class} does not match the original shape {original_shape[0]}")

    def save_json(self, class_averages: dict) -> None:
        """
        Save class averages to a JSON file.

        :param class_averages: Dictionary mapping class labels to class averages.
        :return: None
        """

        with open(self.json_filename, "w") as json_file:
            json.dump(class_averages, json_file, cls=NumpyEncoder)

    def load_json(self) -> dict:
        """
        Load class averages from a JSON file.

        :return: Dictionary mapping class labels to class averages.
        """

        with open(self.json_filename, "r") as json_file:
            class_averages = json.load(json_file)
        return class_averages

    def main(self) -> None:
        """

        :return:
        """

        if self.copy_images:
            self.get_excel_values()

        if not self.load:
            class_averages = {}

            for root, dirs, files in os.walk(self.file_path):
                for class_label in tqdm(dirs, desc=colorama.Fore.BLUE + "Processing classes"):
                    class_coefficients = []

                    class_path = os.path.join(root, class_label)
                    for file in tqdm(os.listdir(class_path), desc=colorama.Fore.YELLOW + "Processing vectors"):
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

            self.plot_euclidean_distances(class_averages)
            self.save_json(class_averages)
        else:
            try:
                class_averages = self.load_json()
                classes = os.listdir(self.images_dir)
                for clas in classes:
                    image_paths = sorted(glob(self.images_dir + f"/{clas}/" + "*"))
                    for idx, image_path in enumerate(image_paths):
                        if idx == 1:
                            self.query_image_path = image_path
                            self.compare_distances(class_averages, self.query_image_path)

            except FileNotFoundError as fne:
                print(f"{fne}")


if __name__ == "__main__":
    try:
        fd = FourierDescriptor(copy_images=False, load=True, order=5)
        fd.main()
    except KeyboardInterrupt:
        print("Ctrl+C pressed")
