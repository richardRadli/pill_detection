import colorama
import cv2
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import openpyxl
import pyefd
import re
import shutil

from tqdm import tqdm
from sklearn.decomposition import PCA

from config.config import ConfigStreamImages
from config.config_selector import dataset_images_path_selector
from utils.utils import create_timestamp, find_latest_file_in_directory, plot_euclidean_distances


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class FourierDescriptor:
    def __init__(self, copy_images, order):
        self.timestamp = create_timestamp()
        colorama.init()

        cfg = ConfigStreamImages().parse()
        self.dataset_name = cfg.dataset_type

        self.order = order
        self.copy_images = copy_images

        self.images_dir = "D:/storage/pill_detection/datasets/cure_two_sided/Reference/stream_images/masks"

        self.file_path = (
            dataset_images_path_selector(self.dataset_name).get("dynamic_margin").get("Fourier_images_by_shape")
        )
        self.plot_euc_dir = (
            dataset_images_path_selector(self.dataset_name).get("dynamic_margin").get("Fourier_euclidean_distance")
        )
        self.json_dir = (
            dataset_images_path_selector(self.dataset_name).get("dynamic_margin").get("Fourier_saved_mean_vectors")
        )
        self.excel_path = (
            os.path.join(
                dataset_images_path_selector(self.dataset_name).get("dynamic_margin").get("pill_desc_xlsx"),
                f"pill_desc_{self.dataset_name}.xlsx"
            )
        )

    def get_excel_values(self) -> None:
        """
        Reads an Excel file, processes the data, and copies corresponding files based on specified conditions.

        :return: None
        """

        workbook = openpyxl.load_workbook(self.excel_path)
        sheet = workbook['Sheet1']

        shape_dict = {}

        for row in sheet.iter_rows(min_row=2, values_only=True):
            pill_id = row[0]
            shape = row[4] if self.dataset_name == "cure_one_sided" else row[6]

            if shape in shape_dict:
                shape_dict[shape].append(pill_id)
            else:
                shape_dict[shape] = [pill_id]

        workbook.close()

        for shape, pill_id_list in tqdm(shape_dict.items(), desc=colorama.Fore.GREEN + "Processing Shapes"):
            logging.info(f"Shape: {shape}")
            for pill_id in pill_id_list:
                src_dir = os.path.join(self.images_dir, str(pill_id))

                try:
                    files = os.listdir(src_dir)
                    for idx, file_name in enumerate(files):
                        if idx == 0:
                            src_file = os.path.join(src_dir, file_name)
                            dst_dir = os.path.join(self.file_path, shape)
                            os.makedirs(dst_dir, exist_ok=True)
                            logging.info(dst_dir)
                            shutil.copy(src_file, dst_dir)
                except FileNotFoundError:
                    logging.error(f"File not found in directory: {src_dir}")

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

    @staticmethod
    def get_largest_contour(segmented_image: np.ndarray):
        """

        :param segmented_image:
        :return:
        """

        contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        return np.squeeze(largest_contour)

    def elliptic_fourier_w_norm(self, segmented_image: np.ndarray):
        """

        :param segmented_image:
        :return:
        """

        largest_contour = self.get_largest_contour(segmented_image)
        coefficients = pyefd.elliptic_fourier_descriptors(largest_contour, order=self.order, normalize=True)
        np.testing.assert_almost_equal(coefficients[0, 0], 1.0, decimal=14)
        np.testing.assert_almost_equal(coefficients[0, 1], 0.0, decimal=14)
        np.testing.assert_almost_equal(coefficients[0, 2], 0.0, decimal=14)
        return coefficients.flatten()[3:]

    @staticmethod
    def plot_vectors(class_averages) -> None:
        """

        Args:
            class_averages:

        Returns:
            None
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

    def save_json(self, dict_to_save: dict, file_name: str) -> None:
        """
        Save class averages to a JSON file.

        Args:
            dict_to_save: Dictionary mapping class labels to class averages.
            file_name: Name of the file to save.

        Returns:
             None
        """

        file_name = os.path.join(self.json_dir, f"{self.timestamp}_%s_{self.order}.json" % file_name)

        with open(file_name, "w") as json_file:
            json.dump(dict_to_save, json_file, cls=NumpyEncoder)

    def load_json(self) -> dict:
        """
        Load class averages from a JSON file.

        Returns:
             Dictionary mapping class labels to class averages.
        """

        latest_file = find_latest_file_in_directory(path=self.json_dir, extension="json")
        with open(latest_file, "r") as json_file:
            class_averages = json.load(json_file)
        return class_averages

    def main(self) -> None:
        if self.copy_images:
            self.get_excel_values()

        class_averages = {}
        pill_coeffs = {}

        for root, dirs, files in os.walk(self.file_path):
            for class_label in tqdm(dirs, desc=colorama.Fore.BLUE + "Processing classes"):
                class_coefficients = []

                class_path = str(os.path.join(root, class_label))
                for file in tqdm(os.listdir(class_path), desc=colorama.Fore.YELLOW + "Processing vectors"):
                    image_path = os.path.join(class_path, file)
                    image_original = cv2.imread(image_path, 1)
                    try:
                        segmented_image = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
                        norm_fourier_coeff = self.elliptic_fourier_w_norm(segmented_image=segmented_image)
                        class_coefficients.append(norm_fourier_coeff)

                        if self.dataset_name == "cure_one_sided":
                            file_key = f"{file.split('_')[0]}_{file.split('_')[2].split('.')[0]}"

                        elif self.dataset_name == "cure_two_sided":
                            file_key = f"{file.split('_')[0]}"

                        elif self.dataset_name == "ogyei":
                            if "_s_" in file:
                                match = re.search(r'^(.*?)_s_\d{3}\.jpg$', file)
                            elif "_u_" in file:
                                match = re.search(r'^(.*?)_u_\d{3}\.jpg$', file)
                            else:
                                match = None

                            if match:
                                file_key = match.group(1)
                            else:
                                raise ValueError(f"Wrong file name: {file}")
                        else:
                            raise ValueError(f"Wrong dataset")

                        if file_key in pill_coeffs:
                            pill_coeffs[file_key].append(norm_fourier_coeff)
                        else:
                            pill_coeffs[file_key] = [norm_fourier_coeff]

                    except Exception as e:
                        logging.error(f"{e}, Could not open{image_path}")

                if class_coefficients:
                    class_average = np.mean(class_coefficients, axis=0)
                    class_averages[class_label] = class_average

        pill_coeffs = dict(sorted(pill_coeffs.items()))
        filename = os.path.join(self.plot_euc_dir, f"euclidean_distances_{self.order}_{self.timestamp}.png")
        plot_euclidean_distances(vectors=class_averages,
                                 dataset_name=self.dataset_name,
                                 filename=filename,
                                 normalize=False,
                                 operation="shapes",
                                 plot_size=8)
        self.plot_vectors(class_averages)
        self.save_json(pill_coeffs, file_name="pill_coeffs_order")


if __name__ == "__main__":
    try:
        fd = FourierDescriptor(copy_images=True, order=10)
        fd.main()
    except KeyboardInterrupt:
        logging.error("Ctrl+C pressed")
