import cv2
import os
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from config.const import CONST


class UnDistortTestImages:
    def __init__(self):
        cam_mtx_np_file = os.path.join(CONST.dir_cam_data, os.listdir(CONST.dir_cam_data)[1])
        data = np.load(cam_mtx_np_file, allow_pickle=True)

        self.matrix = data.item()['matrix']
        self.dist_coeff = data.item()['dist_coeff']
        self.undst_matrix = data.item()['undst_matrix']
        self.roi = data.item()['roi']

    def process_image(self, img_path: str, output_path: str) -> None:
        src_img = cv2.imread(img_path)

        undistorted_image = cv2.undistort(src_img, self.matrix, self.dist_coeff, None, self.undst_matrix)
        x, y, w, h = self.roi
        undistorted_image = undistorted_image[y:y + h, x:x + w]

        cv2.imwrite(output_path, undistorted_image)

    def undistort_images(self) -> None:
        input_dir = "D:/project/IVM/captured_OGYEI_pill_photos_v4"
        output_dir = "D:/project/IVM/captured_OGYEI_pill_photos_v4_undistorted"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with ThreadPoolExecutor() as executor:
            for subdir in tqdm(os.listdir(input_dir)):
                sub_input_dir = os.path.join(input_dir, subdir)
                sub_output_dir = os.path.join(output_dir, subdir)

                if not os.path.exists(sub_output_dir):
                    os.makedirs(sub_output_dir)

                image_paths = [os.path.join(sub_input_dir, filename) for filename in os.listdir(sub_input_dir)]
                output_paths = [os.path.join(sub_output_dir, os.path.basename(path)) for path in image_paths]
                executor.map(self.process_image, image_paths, output_paths)


if __name__ == "__main__":
    undistort_test_images = UnDistortTestImages()
    undistort_test_images.undistort_images()
