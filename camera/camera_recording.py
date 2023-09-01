import cv2
import json
import logging
import os
import pandas as pd
import tkinter as tk

from tkinter import OptionMenu


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++ I M A G E   R E C O R D I N G   G U I  +++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ImageRecordingGUI:
    def __init__(self, root):
        root.title("Image Recording GUI")

        self.pill_names = self.load_pill_names()
        self.selected_pill_name = tk.StringVar(root)
        self.selected_pill_name.set(self.pill_names[0])

        self.pill_name_label = tk.Label(root, text="Pill Name:")
        self.pill_name_label.pack()

        self.pill_name_option_menu = OptionMenu(root, self.selected_pill_name, *self.pill_names)
        self.pill_name_option_menu.pack()

        self.lamp_mode_label = tk.Label(root, text="Lamp Mode:")
        self.lamp_mode_label.pack()
        self.lamp_mode_var = tk.StringVar(root)
        self.lamp_mode_var.set("upper")
        self.lamp_mode_option_menu = tk.OptionMenu(root, self.lamp_mode_var, "upper", "side")
        self.lamp_mode_option_menu.pack()

        self.start_button = tk.Button(root, text="Start Image Capture", command=self.start_capture)
        self.start_button.pack()

        self.set_camera_settings_button = tk.Button(root, text="Set Camera Settings", command=self.set_camera_settings)
        self.set_camera_settings_button.pack()

        self.set_quit_button = tk.Button(root, text="Exit", command=self.exit_program)
        self.set_quit_button.pack()

    @staticmethod
    def load_pill_names():
        try:
            data = pd.read_excel('C:/Users/ricsi/Downloads/pill_names.xlsx', header=None)
            pill_names = data.iloc[:, 0].dropna().tolist()
            return pill_names
        except Exception as e:
            logging.error("Error loading pill names:", e)
            return []

    def start_capture(self):
        pill_name = self.selected_pill_name.get()
        lamp_mode = self.lamp_mode_var.get()

        img_rec = ImageRecording(pill_name=pill_name, lamp_mode=lamp_mode)
        img_rec.capture_images()

    def set_camera_settings(self):
        pill_name = self.selected_pill_name.get()
        lamp_mode = self.lamp_mode_var.get()

        img_rec = ImageRecording(pill_name=pill_name, lamp_mode=lamp_mode)
        img_rec.set_camera_settings()

    @staticmethod
    def exit_program():
        quit()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++ I M A G E   R E C O R D I N G ++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++b
class ImageRecording:
    def __init__(self, pill_name, lamp_mode):
        root_path = "C:/Users/ricsi/Desktop/cam"

        self.camera_data_path = os.path.join(root_path, "camera_data")
        os.makedirs(self.camera_data_path, exist_ok=True)

        self.pill_name = pill_name
        self.image_save_path = os.path.join(root_path, "images/captured_OGYEI_pill_photos_v4", self.pill_name)

        self.coefficient = 1
        self.lamp_mode = lamp_mode
        self.CAM_ID = 0

    # -----------------------------------------------------------------------------------------------------------------
    # --------------------------------- S A V E   C A M   P A R A M S   T O   F I L E ---------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def save_camera_params_to_file(self, filename):
        # Set the camera
        cap = cv2.VideoCapture(self.CAM_ID, cv2.CAP_DSHOW)
        if not (cap.isOpened()):
            raise ValueError("Could not open camera device")

        width = 1280
        height = 720

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        cap.set(cv2.CAP_PROP_SETTINGS, 0)

        while True:
            _, frame = cap.read()
            resized = frame.copy()
            cv2.imshow('Frame', cv2.resize(resized, (width // self.coefficient, height // self.coefficient)))
            key = cv2.waitKey(1)

            if key == ord("q"):
                break
            elif key == ord("s"):
                # Get the current camera parameters
                params = {
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "exposure": int(cap.get(cv2.CAP_PROP_EXPOSURE)),
                    "brightness": int(cap.get(cv2.CAP_PROP_BRIGHTNESS)),
                    "contrast": int(cap.get(cv2.CAP_PROP_CONTRAST)),
                    "saturation": int(cap.get(cv2.CAP_PROP_SATURATION)),
                    "white_balance_blue_u": int(cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)),
                    "white_balance_red_v": int(cap.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V)),
                }

                with open(filename, "w") as f:
                    json.dump(params, f)
                temp_file_name = filename.split("\\")[-1]
                logging.info(f"Camera parameters saved to {temp_file_name}")
                break

        cap.release()
        cv2.destroyAllWindows()

    # -----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- L O A D   C A M   P A R A M S -----------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def load_camera_params_from_file(filename, cam_id=0):
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not (cap.isOpened()):
            raise ValueError("Could not open camera device")

        with open(filename, "r") as f:
            camera_params = json.load(f)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_params["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_params["height"])
        cap.set(cv2.CAP_PROP_EXPOSURE, camera_params["exposure"])
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, camera_params["white_balance_blue_u"])
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, camera_params["white_balance_red_v"])
        cap.set(cv2.CAP_PROP_SATURATION, camera_params["saturation"])
        cap.set(cv2.CAP_PROP_CONTRAST, camera_params["contrast"])
        cap.set(cv2.CAP_PROP_BRIGHTNESS, camera_params["brightness"])

        return cap, camera_params

    # -----------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- T A K E   I M A G E S ---------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def take_images(self, filename):
        cap, camera_params = self.load_camera_params_from_file(filename)

        width = camera_params["width"]
        height = camera_params["height"]

        os.makedirs(self.image_save_path, exist_ok=True)  # Create the directory here
        i = len(os.listdir(self.image_save_path))

        while True:
            _, frame = cap.read()
            copy_image = frame.copy()

            cv2.imshow('Frame', cv2.resize(copy_image, (width // self.coefficient, height // self.coefficient)))
            key = cv2.waitKey(1)

            addition_name = "u" if self.lamp_mode == "upper" else "s"

            if key == ord("q"):
                break
            elif key == ord("c"):
                filename = self.pill_name + "_" + addition_name + "_{:03d}.png".format(i)
                path_to_save = os.path.join(self.image_save_path, filename)
                cv2.imwrite(path_to_save, frame)
                i += 1
                logging.info(f"{filename} image has been captured and saved!")

        cap.release()
        cv2.destroyAllWindows()

    def get_lamp_name(self) -> dict:
        lamp_name = {
            "upper": os.path.join(self.camera_data_path, "camera_params_upper_lamp.json"),
            "side": os.path.join(self.camera_data_path, "camera_params_side_lamp.json")
        }

        return lamp_name

    # -----------------------------------------------------------------------------------------------------------------
    # ------------------------------------- S E T   C A M E R A   S E T T I N G S -------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def set_camera_settings(self):
        filename = self.get_lamp_name()
        self.save_camera_params_to_file(filename.get(self.lamp_mode))

    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- M A I N ----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def capture_images(self):
        filename = self.get_lamp_name()
        self.take_images(filename.get(self.lamp_mode))


if __name__ == "__main__":
    root_gui = tk.Tk()
    app = ImageRecordingGUI(root_gui)
    root_gui.mainloop()
