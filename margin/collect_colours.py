import cv2
import json
import numpy as np
import os

from glob import glob

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses an input image by applying edge detection, thresholding, dilation, and connected components
    analysis.

    :param image: Input image.
    :return: Preprocessed image.
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    return pill_mask


def extract_colour(image, width, height, x, y):
    roi = image[y:y + height, x:x + width]
    average_color = np.mean(roi, axis=(0, 1))
    average_color_rgb = average_color[::-1]
    return [int(avg_color) for avg_color in average_color_rgb]


def plot_image(image, width, height, x1, y1, x2, y2, filename):
    cv2.rectangle(image, (x1, y1), (x1 + width, y1 + height), (0, 0, 255), 2)
    cv2.rectangle(image, (x2, y2), (x2 + width, y2 + height), (0, 0, 255), 2)
    cv2.imwrite(f"C:/Users/ricsi/Desktop/color_pills/annotated/{filename}", image)


def main():
    cfg = ConfigAugmentation().parse()
    images_dir = (
        dataset_images_path_selector(cfg.dataset_name).get("src_stream_images").get("reference").get("stream_images_rgb")
    )

    width, height = 30, 30
    classes = os.listdir(images_dir)

    shape_class_colour = {}

    for class_name in classes:
        image_paths = sorted(glob(images_dir + f"/{class_name}/" + "*"))
        for idx, image_path in enumerate(image_paths):
            if idx == 1:
                image = cv2.imread(image_path)
                binary_image = preprocess_image(image)
                non_zero_pixels = cv2.findNonZero(binary_image)

                y_coordinates = non_zero_pixels[:, 0, 1]
                top_y = np.min(y_coordinates)
                down_y = np.max(y_coordinates)
                x1, x2, y1, y2 = image.shape[1] // 2, image.shape[1] // 2, top_y + 20, down_y - 40

                # x_coordinates = non_zero_pixels[:, 0, 0]
                # leftmost_x = np.min(x_coordinates)
                # rightmost_x = np.max(x_coordinates)
                # x1, x2, y1, y2 = leftmost_x+20, rightmost_x-40, image.shape[0]//2, image.shape[0]//2,

                rgb1 = extract_colour(image, width, height, x1, y1)
                rgb2 = extract_colour(image, width, height, x2, y2)
                print("Average Color (RGB1):", rgb1)
                print("Average Color (RGB2):", rgb2)
                filename = os.path.basename(image_path)
                plot_image(image, width, height, x1, y1, x2, y2, filename)
                shape_class_colour[class_name] = (rgb1, rgb2)

    sorted_dict = dict(sorted(shape_class_colour.items(), key=lambda item: int(item[0])))

    with open("C:/Users/ricsi/Desktop/color_pills/colors_cure.json", "w") as json_file:
        json.dump(sorted_dict, json_file, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
