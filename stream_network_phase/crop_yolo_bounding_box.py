import cv2
import os

from tqdm import tqdm

from utils.utils import file_reader


def save_annotations(image_path, annotation_path, out_path):
    images = file_reader(image_path, "png")

    for img in tqdm(images, total=len(images)):
        # Get the corresponding annotation file
        img_name = os.path.splitext(os.path.basename(img))[0]
        ann = os.path.join(annotation_path, img_name + ".txt")

        if os.path.isfile(ann):
            image = cv2.imread(img)
            height, width, _ = image.shape

            with open(ann, 'r') as f:
                lines = f.readlines()

            # Parse annotation lines and extract relevant information
            annotations = [line.strip().split() for line in lines]
            annotations = [
                [int(annotation[0]), float(annotation[1]), float(annotation[2]), float(annotation[3]),
                 float(annotation[4])] for annotation in annotations]

            # Find the smallest bounding box
            smallest_box = min(annotations, key=lambda box: box[3] * box[4])

            _, x_center, y_center, box_width, box_height = smallest_box

            # Convert relative coordinates to absolute coordinates
            x_min = int((x_center - box_width / 2) * width)
            y_min = int((y_center - box_height / 2) * height)
            x_max = int((x_center + box_width / 2) * width)
            y_max = int((y_center + box_height / 2) * height)

            # Calculate the longer side of the bounding box
            box_size = max(x_max - x_min, y_max - y_min)

            # Calculate the center coordinates of the square bounding box
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            # Calculate the new coordinates of the square bounding box
            x_min = center_x - box_size // 2
            y_min = center_y - box_size // 2
            x_max = center_x + box_size // 2
            y_max = center_y + box_size // 2

            # Calculate the crop coordinates
            bounding_pixels = 10
            left = x_min - bounding_pixels
            upper = y_min - bounding_pixels
            right = x_max + bounding_pixels
            lower = y_max + bounding_pixels

            # Crop the image using NumPy array indexing
            cropped_image_array = image[upper:lower, left:right]
            cv2.imwrite(os.path.join(out_path, img_name + ".png"), cropped_image_array)
        else:
            print(f"No annotation file found for image: {os.path.basename(img)}")


if __name__ == "__main__":
    img_path = "C:/Users/ricsi/Documents/yolov7/runs/detect/ogyeiv2_binary_class/"
    ann_path = "C:/Users/ricsi/Documents/yolov7/runs/detect/ogyeiv2_binary_class/labels/"
    dst_path = "C:/Users/ricsi/Desktop/crops/"
    save_annotations(img_path, ann_path, dst_path)
