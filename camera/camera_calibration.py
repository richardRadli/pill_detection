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

from config.config import CameraAndCalibrationConfig
from config.config_selector import camera_config
from utils.utils import find_latest_subdir, setup_logger

np.set_printoptions(precision=6, suppress=True)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++++++++++++++++++++++++++++++++++++++++ C A M E R A   C A L I B R A T I O N ++++++++++++++++++++++++++++++++++++++ #
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class CameraCalibration:
    def __init__(self, undistortion_needed):
        """
        CameraCalibration class
        ---------------------------

        Parameters
        -----------
        undistortion_needed: :class:`bool`
            undistortion either needed or not
        """

        setup_logger()
        self.cam_cfg = CameraAndCalibrationConfig().parse()

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.cam_cfg.square_size, 0.001)
        self.matrix = np.empty([3, 3])
        self.undistorted_matrix = np.empty([3, 3])
        self.dist_coefficients = np.empty([1, 5])
        self.roi = None
        self.rot_vecs = None
        self.trs_vecs = None
        self.obj_pts = []
        self.img_pts = []
        self.chs_col = self.cam_cfg.chs_col
        self.chs_row = self.cam_cfg.chs_row
        self.img_size = None
        self.original_images = []

        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        source_images = find_latest_subdir(camera_config().get("calibration_images"))

        self.orig_img_list = sorted(glob(f'{source_images}/*.jpg') + glob(f'{source_images}/*.png'))
        self.undistorted_images_path = os.path.join(camera_config().get("undistorted_images"), self.timestamp)
        os.makedirs(self.undistorted_images_path, exist_ok=True)

        self.save_data_path = os.path.join(camera_config().get("camera_matrix"), self.timestamp)
        os.makedirs(self.save_data_path, exist_ok=True)

        self.undistortion_needed = undistortion_needed

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
    def calibration(self) -> None:
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
                    self.original_images.append(img_data)
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
        rms, self.matrix, self.dist_coefficients, self.rot_vecs, self.trs_vecs = (
            cv2.calibrateCamera(objectPoints=self.obj_pts,
                                imagePoints=self.img_pts,
                                imageSize=self.img_size,
                                cameraMatrix=None,
                                distCoeffs=None)
        )

        logging.info(f'Root mean square error\n{rms}')
        logging.info(f'Camera matrix\n{self.matrix}')
        logging.info(f'Distortion coefficients\n{self.dist_coefficients}')
        logging.info('-----C A L I B R A T I O N   F I N I S H E D-----')

    # ---------------------------------------------------------------------------------------------------------------- #
    # --------------------------------------- U N D I S T O R T   A N D   S A V E ------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def undistort_and_save(name: str, image, matrix, distortion_coefficients, undistorted_matrix, roi,
                           undistorted_images_path: str) -> None:
        """Undistorts the image using the required parameters before saving.

        Parameters
        -----------
        name: :class:`str`
            The original name of the image
        image:
            The image data
        matrix:
            The camera matrix
        distortion_coefficients:
            The distortion coefficient
        undistorted_matrix:
            The undistorted camera matrix
        roi:
            The region of interest
        undistorted_images_path: :class:`str`
            The path to use when saving the undistorted image
        """

        image = cv2.undistort(image, matrix, distortion_coefficients, None, undistorted_matrix)
        x, y, w, h = roi
        image = image[y:y + h, x:x + w]
        path = os.path.join(undistorted_images_path, os.path.basename(name))
        cv2.imwrite(path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    # ---------------------------------------------------------------------------------------------------------------- #
    # ------------------------------- G E N E R A T E   N E W   C A M E R A   M A T R I X ---------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def generate_new_camera_matrix(self) -> None:
        """Calculates the undistorted camera matrix and the region of interest for undistorted images.
        Raises
        -------
        ``AssertionError`` if calibration error is over the given error_threshold value.
        """

        logging.info('----- O B T A I N I N G   N E W   C A M E R A   M A T R I X-----')

        name, img = self.original_images[0]
        h, w = img.shape[:2]
        self.undistorted_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.matrix, self.dist_coefficients, (w, h), 1, (w, h))

        logging.info('New camera matrix')
        logging.info(str(self.undistorted_matrix))
        self.undistort_and_save(name, img, self.matrix, self.dist_coefficients, self.undistorted_matrix, self.roi, self.undistorted_images_path)

        mean_error = 0
        for i in range(len(self.obj_pts)):
            img_points2, _ = cv2.projectPoints(self.obj_pts[i], self.rot_vecs[i], self.trs_vecs[i], self.matrix,
                                               self.dist_coefficients)
            error = cv2.norm(self.img_pts[i], img_points2, cv2.NORM_L2) / len(img_points2)
            mean_error += error
        total_err = mean_error / len(self.obj_pts)
        logging.info(f'Total error: {total_err}')

        assert total_err < self.cam_cfg.error_threshold, 'Camera calibration error too high!'
        path = os.path.join(self.save_data_path, "undistorted_cam_mtx.npy")
        np.save(path, {'matrix': self.matrix,
                       'distortion_coefficients': self.dist_coefficients,
                       'undistorted_matrix': self.undistorted_matrix,
                       'roi': self.roi})

    # ---------------------------------------------------------------------------------------------------------------- #
    # ----------------------------------------- U N D I S T O R T   I M A G E S -------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def undistort_images(self) -> None:
        """
        Undistorts every image used during calibration and saves them in the appropriate folder.
        """

        start = perf_counter()

        with ThreadPoolExecutor() as executor:
            futures = []
            for name, img in self.original_images[1:]:
                futures.append(executor.submit(self.undistort_and_save, name, img, self.matrix, self.dist_coefficients,
                                               self.undistorted_matrix, self.roi, self.undistorted_images_path))

        logging.info(f'Undistortion time: {perf_counter() - start} s')

    # ---------------------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------------- M A I N -------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def main(self):
        self.calibration()
        self.generate_new_camera_matrix()
        if self.undistortion_needed:
            self.undistort_images()


# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------- _ _ M A I N _ _ ------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    try:
        cal = CameraCalibration(undistortion_needed=True)
        cal.main()
    except KeyboardInterrupt as kie:
        logging.error(f"{kie}")
