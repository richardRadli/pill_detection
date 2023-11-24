import cv2
import os

from datetime import datetime


class CalibrationImageCapture:
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.y_max = None
        self.x_max = None
        self.y_min = None
        self.x_min = None
        self.cap = None
        self.capture_count = 0
        self.crop_size = 1600
        self.root_dir = "C:/Users/ricsi/Desktop/cam/"
        self.setup_camera()

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Could not open camera device")
            exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 2500)

        self.x_min = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) - self.crop_size) // 2
        self.y_min = (self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - self.crop_size) // 2
        self.x_max = self.x_min + self.crop_size
        self.y_max = self.y_min + self.crop_size

    def capture_images(self):
        location = os.path.join(self.root_dir, "images", "camera_calibration_images", f"{self.timestamp}")
        os.makedirs(location, exist_ok=True)
        os.chdir(location)

        while True:
            ret, frame = self.cap.read()
            resized_frame = frame.copy()[int(self.y_min):int(self.y_max), int(self.x_min):int(self.x_max)]

            cv2.imshow('Calibration Image', resized_frame)
            key = cv2.waitKey(1)

            if key == ord("q"):
                break
            elif key == ord("c"):
                filename = f"{self.capture_count:03d}.png"
                cv2.imwrite(filename, frame)
                print("Captured image:", filename)
                self.capture_count += 1

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    calibration = CalibrationImageCapture()
    calibration.capture_images()
