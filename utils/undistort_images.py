import os
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm


class UnDistortTestImages:
    def __init__(self):
        self.loc_to_save = "C:/Users/ricsi/Desktop/undistorted_images"
        self.loc_of_undis_test_imgs = "C:/Users/ricsi/Desktop/images"
        cam_mtx_np_file = os.path.join("2023-03-27_12-45-45_undistorted_cam_mtx.npy")
        data = np.load(cam_mtx_np_file, allow_pickle=True)

        self.matrix = data.item()['matrix']
        self.dist_coeff = data.item()['dist_coeff']
        self.undst_matrix = data.item()['undst_matrix']
        self.roi = data.item()['roi']

        self.crop_size = 800

        os.makedirs(self.loc_to_save, exist_ok=True)

    @staticmethod
    def crop_image(height, width, crop_size):
        x_min = int(width / 2 - (crop_size / 2))
        y_min = int(height / 2 - (crop_size / 2))
        x_max = int(width / 2 + (crop_size / 2))
        y_max = int(height / 2 + (crop_size / 2))

        return x_min, y_min, x_max, y_max

    def undistort_images(self):
        images = sorted(glob(self.loc_of_undis_test_imgs + "/*.png"))

        for idx, name in tqdm(enumerate(images), total=len(images)):
            src_img = cv2.imread(name)
            file_name = os.path.basename(name)
            save_path = os.path.join(self.loc_to_save, file_name)

            undistorted_image = cv2.undistort(src_img, self.matrix, self.dist_coeff, None, self.undst_matrix)
            x, y, w, h = self.roi
            undistorted_image = undistorted_image[y:y + h, x:x + w]

            x_min, y_min, x_max, y_max = self.crop_image(height=undistorted_image.shape[0],
                                                         width=undistorted_image.shape[1],
                                                         crop_size=self.crop_size)
            crop_img = undistorted_image[y_min:y_max, x_min:x_max]

            cv2.imwrite(save_path, crop_img)


if __name__ == "__main__":
    undistort_test_images = UnDistortTestImages()
    undistort_test_images.undistort_images()
