import cv2
import logging
import os

from datetime import datetime

from config.config import CameraAndCalibrationConfig
from config.config_selector import camera_config
from utils.utils import setup_logger


class CalibrationImageCapture:
    def __init__(self):
        setup_logger()
        self.cam_cfg = CameraAndCalibrationConfig().parse()
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.cap = None
        self.capture_count = 0
        self.size_coefficients = self.cam_cfg.size_coefficients
        self.root_dir = camera_config().get("calibration_images")

        self.setup_camera()

    def setup_camera(self) -> None:
        """
        This function tries to open the camera, and set the parameters of the device.
        :return: None
        """

        self.cap = cv2.VideoCapture(self.cam_cfg.cam_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logging.error("Could not open camera device")
            exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_cfg.height)
        self.cap.set(cv2.CAP_PROP_SETTINGS, 0)

    def capture_images(self) -> None:
        """
        This function opens the stream, displays it on a window. If [c] is pressed, we save an image to a designated
        directory. If button [q] is pressed, the program shuts down.
        :return: None
        """

        location = os.path.join(self.root_dir, f"{self.timestamp}")
        os.makedirs(location, exist_ok=True)

        while True:
            _, frame = self.cap.read()
            resized_frame = cv2.resize(frame, dsize=(frame.shape[1] // self.size_coefficients,
                                                     frame.shape[0] // self.size_coefficients))

            cv2.imshow('Calibration Image', resized_frame)
            key = cv2.waitKey(1)

            if key == ord("q"):
                break
            elif key == ord("c"):
                filename = f"calibration_image_{self.capture_count:03d}.png"
                filename_location = os.path.join(location, filename)
                cv2.imwrite(filename_location, frame)
                logging.info(f"Captured image: {filename}")
                self.capture_count += 1

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    calibration = CalibrationImageCapture()
    calibration.capture_images()
