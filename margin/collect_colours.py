import cv2
import json
import os

from glob import glob

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector
from utils.utils import create_timestamp


# Function to handle mouse events
def click_event(event, x, y, flags, param):
    global points, sampling, image, image_copy, num_selections, save_point
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        num_selections += 1
        if num_selections == 2:
            save_point = True
    elif event == cv2.EVENT_LBUTTONUP and save_point:
        roi = image_copy[points[0][1]:points[1][1], points[0][0]:points[1][0]]
        average_color = [int(roi[:, :, i].mean()) for i in range(3)]
        data[folder_name][os.path.basename(image_path)]\
            ["p" + str(len(data[folder_name][os.path.basename(image_path)]) + 1)] = average_color
        cv2.rectangle(image, points[0], points[1], (0, 0, 255))

        cv2.imshow('image', image)
        cv2.imshow('ROI', roi)
        cv2.waitKey(0)
        cv2.destroyWindow('ROI')
        num_selections = 0
        points.clear()
        save_point = False


cfg = ConfigAugmentation().parse()

timestamp = create_timestamp()

images_dir = (
    dataset_images_path_selector("ogyei").get("src_stream_images").get("reference").get("stream_images_rgb")
)

# Load the image
for directory in sorted(glob(os.path.join(images_dir, "*"))):
    for image_path in sorted(glob(directory + "/*")):
        image = cv2.imread(image_path)
        image_copy = image.copy()

        # Initialize variables
        points = []
        sampling = True
        num_selections = 0
        save_point = False
        data = {}
        folder_name = os.path.basename(directory)
        data[folder_name] = {os.path.basename(image_path): {}}

        # Create a window and set the mouse callback function
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', click_event)

        # Main loop
        while sampling:
            cv2.imshow('image', image_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                sampling = False

        cv2.destroyAllWindows()

        # Save data to JSON after iterating through all images
        json_save_dir = (
            os.path.join(dataset_images_path_selector(cfg.dataset_name).get("dynamic_margin").get("colour_vectors"),
                         timestamp)
        )
        os.makedirs(json_save_dir, exist_ok=True)
        json_path = os.path.join(json_save_dir,
                                 os.path.basename(image_path).replace(".jpg", ".json"))
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        print("Data saved to:", json_path)
