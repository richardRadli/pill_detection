import cv2
import logging
import os
from datetime import datetime

from utils.utils import setup_logger


class CalibrationImageCapture:
    def __init__(self):
        setup_logger()
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.cap = None
        self.capture_count = 0
        self.size_coeff = 3
        self.root_dir = "C:/Users/ricsi/Desktop/cam"
        self.setup_camera()

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logging.error("Could not open camera device")
            exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3479)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2159)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)

    def capture_images(self):
        location = os.path.join(self.root_dir, "camera_calibration_images", f"{self.timestamp}")
        os.makedirs(location, exist_ok=True)

        while True:
            _, frame = self.cap.read()
            resized_frame = cv2.resize(frame, dsize=(frame.shape[1] // self.size_coeff,
                                                     frame.shape[0] // self.size_coeff))

            cv2.imshow('Calibration Image', resized_frame)
            key = cv2.waitKey(1)

            if key == ord("q"):
                break
            elif key == ord("c"):
                filename = f"{self.capture_count:03d}.png"
                filename_location = os.path.join(location, filename)
                cv2.imwrite(filename_location, frame)
                logging.info(f"Captured image: {filename}")
                self.capture_count += 1

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    calibration = CalibrationImageCapture()
    calibration.capture_images()