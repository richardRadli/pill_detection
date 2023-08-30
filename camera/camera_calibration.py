__author__ = 'Richárd Rádli and Attila Raschek'
__version__ = '1.3.0'
__maintainer__ = 'Richárd Rádli and Attila Raschek'
__email__ = 'radli.richard@mik.uni-pannon.hu'
__status__ = 'Production'

import cv2
import logging
import numpy as np
import os

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from time import perf_counter
from tqdm import tqdm

from utils.utils import setup_logger

np.set_printoptions(precision=6, suppress=True)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++++++++++++++++++++++++++++++++++++++++ C A M E R A   C A L I B R A T I O N ++++++++++++++++++++++++++++++++++++++ #
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class CameraCalibration:
    def __init__(self, undist_needed):
        """
        CameraCalibration class
        ---------------------------

        Parameters
        -----------
        undist_needed: :class:`bool`
            undistortion needed or not
        """

        setup_logger()

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)
        self.matrix = np.empty([3, 3])
        self.undst_matrix = np.empty([3, 3])
        self.dist_coeff = np.empty([1, 5])
        self.roi = None
        self.rot_vecs = None
        self.trs_vecs = None
        self.obj_pts = []
        self.img_pts = []
        self.chs_col = 7
        self.chs_row = 6
        self.img_size = None
        self.orig_imgs = []

        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        root_dir = "C:/Projects/IVM_gyogyszer"
        undistorted_images = os.path.join(root_dir, "calibration/2023-07-04")

        self.orig_img_list = sorted(glob(f'{undistorted_images}/*.jpg'))
        self.undist_path = os.path.join(root_dir, "images/undist/")
        os.makedirs(self.undist_path, exist_ok=True)

        self.save_data_path = os.path.join(root_dir, "camera_data/")
        os.makedirs(self.save_data_path, exist_ok=True)

        self.undist_needed = undist_needed

    # ---------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------------- F I N D   C O R N E R S ------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def find_corners(self, thread_id: int, image_name: str):
        """
        Finds chessboard corner subpixels on an image.

        Parameters
        -----------
        thread_id: :class:`int`
            The numerical ID of the thread, for easy identification and ordering of results
        image_name: :class:`str`
            The path of the image
        """
        name = f'thr-{thread_id}'
        corners_result = None

        src = cv2.imread(image_name)
        image_name = os.path.basename(image_name)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        if self.img_size is None:
            self.img_size = gray.shape[:2]

        ret, corners = cv2.findChessboardCorners(gray, (self.chs_col, self.chs_row), None, cv2.CALIB_CB_FAST_CHECK)
        image_data = (image_name, src)

        if ret:
            corners_result = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        else:
            logging.warning(f'{name}: Chessboard corners could not be found {image_name}')

        return corners_result, image_data

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------- C A L I B R A T I O N ------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def calibration(self):
        """
        Calculates the intrinsic camera matrix and the distortion coefficient.
        """

        accepted = 0
        start = perf_counter()

        with ThreadPoolExecutor() as executor:
            futures = []
            for i, img in tqdm(enumerate(self.orig_img_list), total=len(self.orig_img_list), desc="Collecting images"):
                futures.append(executor.submit(self.find_corners, i, img))
            try:
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing calibration images"):
                    corners, img_data = future.result()
                    if corners is not None:
                        accepted += 1
                        self.img_pts.append(corners)
                    self.orig_imgs.append(img_data)
            except Exception as e:
                logging.error(str(e))

        logging.info(f'Execution time: {perf_counter() - start} s')
        rejected = len(self.orig_img_list) - accepted
        logging.info(f'Accepted images: {accepted}')
        logging.info(f'Rejected images: {rejected}')

        objp = np.zeros((self.chs_row * self.chs_col, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chs_col, 0:self.chs_row].T.reshape(-1, 2)
        self.obj_pts = [objp] * len(self.img_pts)

        logging.info('------C A L I B R A T I O N   S T A R T E D-----')
        rms, self.matrix, self.dist_coeff, self.rot_vecs, self.trs_vecs = cv2.calibrateCamera(self.obj_pts,
                                                                                              self.img_pts,
                                                                                              self.img_size, None, None)

        logging.info(f'Root mean square error\n{rms}')
        logging.info(f'Camera matrix\n{self.matrix}')
        logging.info(f'Distortion coefficients\n{self.dist_coeff}')
        logging.info('-----C A L I B R A T I O N   F I N I S H E D-----')

    # ---------------------------------------------------------------------------------------------------------------- #
    # ------------------------------- G E N E R A T E   N E W   C A M E R A   M A T R I X ---------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def generate_new_camera_matrix(self, error_threshold: int = 0.2):
        """Calculates the undistorted camera matrix and the region of interest for undistorted images.
        Parameters
        -----------
        error_threshold: :class:`int`
            The threshold which above the program halts.
        Raises
        -------
        ``AssertionError`` if calibration error is over the given error_threshold value.
        """

        logging.info('----- O B T A I N I N G   N E W   C A M E R A   M A T R I X-----')

        name, img = self.orig_imgs[0]
        h, w = img.shape[:2]
        self.undst_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.matrix, self.dist_coeff, (w, h), 1,
                                                                    (w, h))

        logging.info('New camera matrix')
        logging.info(str(self.undst_matrix))
        self.undistort_and_save(name, img, self.matrix, self.dist_coeff, self.undst_matrix, self.roi, self.undist_path)

        mean_error = 0
        for i in range(len(self.obj_pts)):
            img_points2, _ = cv2.projectPoints(self.obj_pts[i], self.rot_vecs[i], self.trs_vecs[i], self.matrix,
                                               self.dist_coeff)
            error = cv2.norm(self.img_pts[i], img_points2, cv2.NORM_L2) / len(img_points2)
            mean_error += error
        total_err = mean_error / len(self.obj_pts)
        logging.info(f'Total error: {total_err}')

        assert total_err < error_threshold, 'Camera calibration error too high!'
        path = os.path.join(self.save_data_path, self.timestamp + "_undistorted_cam_mtx.npy")
        np.save(path, {'matrix': self.matrix, 'dist_coeff': self.dist_coeff, 'undst_matrix': self.undst_matrix,
                       'roi': self.roi})

    # ---------------------------------------------------------------------------------------------------------------- #
    # ----------------------------------------- U N D I S T O R T   I M A G E S -------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def undistort_images(self):
        """
        Undistorts every image used during calibration and saves them in the appropriate folder.
        """

        start = perf_counter()

        with ThreadPoolExecutor() as executor:
            futures = []
            for name, img in self.orig_imgs[1:]:
                futures.append(executor.submit(self.undistort_and_save, name, img, self.matrix, self.dist_coeff,
                                               self.undst_matrix, self.roi, self.undist_path))

        logging.info(f'Undistortion time: {perf_counter() - start} s')

    # ---------------------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------------- M A I N -------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def main(self):
        self.calibration()
        self.generate_new_camera_matrix()
        if self.undist_needed:
            self.undistort_images()

    # ---------------------------------------------------------------------------------------------------------------- #
    # --------------------------------------- U N D I S T O R T   A N D   S A V E ------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def undistort_and_save(name: str, image, matrix, dist_coeff, undst_matrix, roi, undst_path: str):
        """Undistorts the image using the required parameters before saving.

        Parameters
        -----------
        name: :class:`str`
            The original name of the image
        image:
            The image data
        matrix:
            The camera matrix
        dist_coeff:
            The distortion coefficient
        undst_matrix:
            The undistorted camera matrix
        roi:
            The region of interest
        undst_path: :class:`str`
            The path to use when saving the undistorted image
        """

        image = cv2.undistort(image, matrix, dist_coeff, None, undst_matrix)
        x, y, w, h = roi
        image = image[y:y + h, x:x + w]
        path = os.path.join(undst_path, os.path.basename(name))
        cv2.imwrite(path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])


# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------- _ _ M A I N _ _ ------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    cal = CameraCalibration(undist_needed=True)
    cal.main()
