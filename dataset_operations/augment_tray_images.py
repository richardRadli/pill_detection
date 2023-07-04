import cv2
import logging
import numpy as np
import os
import random
import shutil

from config.const import DATASET_PATH
from tqdm import tqdm
from typing import Tuple


class AugmentTrayImages:
    def __init__(self):
        self.root = "C:/Users/ricsi/Desktop/tray"
        self.tray_images = os.path.join(self.root, "2023-07-04")
        self.tray_images_aug = os.path.join(self.root, "tray_aug")
        self.image_on_tray = os.path.join(self.root, "image_on_tray")
        self.image_on_tray_aug = os.path.join(self.root, "image_on_tray_aug")
        self.medicine_images = DATASET_PATH.get_data_path("cure_reference")
        self.medicine_mask = DATASET_PATH.get_data_path("cure_reference_mask")

        # Get the list of all images
        image_files = os.listdir(self.tray_images)
        image_files.sort()

        # Collect the unique classes from the image filenames
        all_classes = set()
        for image_file in image_files:
            class_name = image_file.split('_')[0]
            all_classes.add(class_name)
        self.all_classes = sorted(list(all_classes))

    @staticmethod
    def create_directories(classes, images_dir, dataset_path):
        for class_name in classes:
            class_path = os.path.join(images_dir, class_name)
            os.makedirs(class_path, exist_ok=True)

            images = [image for image in os.listdir(dataset_path) if image.startswith(f"{class_name}_")]
            for image in tqdm(images, total=len(images), desc="Copying images"):
                src_path = os.path.join(dataset_path, image)
                dest_path = os.path.join(class_path, image)
                shutil.copy(src_path, dest_path)

    @staticmethod
    def rename_file(image_path, op):
        directory = os.path.dirname(image_path)
        filename = os.path.basename(image_path)

        name, extension = os.path.splitext(filename)

        new_filename = f"{name}_{op}"
        counter = 1
        final_file_name = os.path.join(directory, f"{new_filename}_{counter}{extension}")

        while os.path.isfile(final_file_name):
            counter += 1
            final_file_name = os.path.join(directory, f"{new_filename}_{counter}{extension}")

        return final_file_name

    @staticmethod
    def copy_original_image(src_path, dst_path):
        shutil.copy(src_path, dst_path)

    def change_white_balance(self, image_path: str, aug_path, domain: Tuple[float, float] = (0.7, 1.2)) -> None:
        image = cv2.imread(image_path)

        # Generate random scaling factors for each color channel
        scale_factors = np.random.uniform(low=domain[0], high=domain[1], size=(3,))

        # Apply the scaling factors to the image
        adjusted_image = image * scale_factors

        # Clip the pixel values to the valid range [0, 255]
        adjusted_image = np.clip(adjusted_image, 0, 255)
        adjusted_image = adjusted_image.astype(np.uint8)

        new_image_file_name = self.rename_file(aug_path, op="distorted_colour")
        cv2.imwrite(new_image_file_name, adjusted_image)

    def gaussian_smooth(self, image_path, aug_path, kernel) -> None:
        image = cv2.imread(image_path)

        smoothed_image = cv2.GaussianBlur(image, kernel, 0)

        new_image_file_name = self.rename_file(aug_path, op="gaussian_%s" % str(kernel[0]))

        cv2.imwrite(new_image_file_name, smoothed_image)

    def change_brightness(self, image_path, aug_path, exposure_factor: float) -> None:
        image = cv2.imread(image_path)

        image = image.astype(np.float32) / 255.0
        adjusted_image = image * exposure_factor
        adjusted_image = np.clip(adjusted_image, 0, 1)
        adjusted_image = (adjusted_image * 255).astype(np.uint8)

        new_image_file_name = self.rename_file(aug_path, op="brightness")
        cv2.imwrite(new_image_file_name, adjusted_image)

    @staticmethod
    def unique_count_app(img):
        img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
        colors, count = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
        return tuple(colors[count.argmax()])

    def rotate_image(self, image_path, aug_path, angle: int) -> None:
        image = cv2.imread(image_path)

        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

        clr = self.unique_count_app(image)
        clr = tuple(value.item() for value in clr)

        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=clr)
        new_image_file_name = self.rename_file(aug_path, op="rotated_%s" % str(angle))
        cv2.imwrite(new_image_file_name, rotated_image)

    def shift_image(self, image_path, aug_path, shift_x: int = 50, shift_y: int = 100):
        image = cv2.imread(image_path)

        height, width = image.shape[:2]

        mtx = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        clr = self.unique_count_app(image)
        clr = tuple(value.item() for value in clr)

        shifted_image = cv2.warpAffine(image, mtx, (width, height), borderValue=clr)
        new_image_file_name = self.rename_file(aug_path, op="shifted")
        cv2.imwrite(new_image_file_name, shifted_image)

    def zoom_in_object(self, image_path, aug_path, crop_size):
        image = cv2.imread(image_path)

        height, width = image.shape[:2]

        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2
        end_x = start_x + crop_size
        end_y = start_y + crop_size

        cropped_image = image[start_y:end_y, start_x:end_x]
        zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)

        new_image_file_name = self.rename_file(aug_path, op="zoomed")
        cv2.imwrite(new_image_file_name, zoomed_image)

    def flip_image(self, image_path, aug_path, flip_direction):
        # Read the image
        image = cv2.imread(image_path)

        # Flip the image based on the specified direction
        if flip_direction == 'horizontal':
            flipped_image = cv2.flip(image, 1)  # Flip horizontally (around the y-axis)
        elif flip_direction == 'vertical':
            flipped_image = cv2.flip(image, 0)  # Flip vertically (around the x-axis)
        else:
            raise ValueError("Invalid flip direction. Must be 'horizontal' or 'vertical'.")

        new_image_file_name = self.rename_file(aug_path, op="flipped_%s" % flip_direction)
        cv2.imwrite(new_image_file_name, flipped_image)

    def place_medicine_on_tray(self, pill_image_path, pill_mask_path, tray_image_path):
        # Read the pill image and its corresponding mask
        pill_image = cv2.imread(pill_image_path)
        pill_mask = cv2.imread(pill_mask_path, cv2.IMREAD_GRAYSCALE)

        # Threshold the mask to create a binary mask
        _, binary_mask = cv2.threshold(pill_mask, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select the largest contour (assuming it corresponds to the pill)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a bounding rectangle around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Extract the region of interest (ROI) from the pill image
        pill_roi = pill_image[y:y + h, x:x + w]

        # Read the tray image
        tray_image = cv2.imread(tray_image_path)

        # Get the dimensions of the tray image
        tray_height, tray_width = tray_image.shape[:2]

        # Resize the pill ROI to fit within the tray image
        resized_pill_roi = cv2.resize(pill_roi, (w, h))

        # Calculate the coordinates to place the pill ROI on the tray image
        x_offset = random.randint(0, tray_width - w)
        y_offset = random.randint(0, tray_height - h)

        # Create a mask for the pill ROI
        pill_mask_roi = binary_mask[y:y + h, x:x + w]

        # Apply the pill ROI on the tray image using the mask
        tray_image[y_offset:y_offset + h, x_offset:x_offset + w] = np.where(
            np.expand_dims(pill_mask_roi, axis=2),
            resized_pill_roi,
            tray_image[y_offset:y_offset + h, x_offset:x_offset + w]
        )

        class_name = (os.path.basename(tray_image_path).split('_')[0])
        out_path_dir = os.path.join(self.image_on_tray, class_name)
        os.makedirs(out_path_dir, exist_ok=True)
        new_file_name = os.path.join(out_path_dir, os.path.basename(tray_image_path))
        cv2.imwrite(new_file_name, tray_image)

    def main(self, create_class_dirs: bool, first_phase, second_phase, third_phase) -> None:
        if create_class_dirs:
            self.create_directories(self.all_classes, self.tray_images, self.tray_images)

        for class_name in self.all_classes:
            class_dir = os.path.join(self.tray_images, class_name)
            class_dir_aug = os.path.join(self.tray_images_aug, class_name)
            class_image_on_tray = os.path.join(self.image_on_tray, class_name)
            class_dir_tray_aug = os.path.join(self.image_on_tray_aug, class_name)

            os.makedirs(class_dir_aug, exist_ok=True)
            os.makedirs(class_dir_tray_aug, exist_ok=True)

            image_files = os.listdir(class_dir)
            images_files_aug = os.listdir(class_dir_aug)
            images_files_aug_tray = os.listdir(class_image_on_tray)

            if first_phase:
                # First phase, augmenting empty tray images
                for _, image_file in tqdm(enumerate(image_files), total=len(image_files)):
                    full_path_image = os.path.join(class_dir, image_file)
                    full_path_image_aug = os.path.join(class_dir_aug, image_file)
                    self.copy_original_image(full_path_image, full_path_image_aug)
                    self.change_white_balance(full_path_image, full_path_image_aug, domain=(0.7, 1.2))
                    self.gaussian_smooth(full_path_image, full_path_image_aug, kernel=(7, 7))
                    self.change_brightness(full_path_image, full_path_image_aug, exposure_factor=random.uniform(0.5, 1.5))
                    self.rotate_image(full_path_image, full_path_image_aug, angle=random.randint(35, 270))
                    self.shift_image(full_path_image, full_path_image_aug, 150, 200)
                    self.flip_image(full_path_image, full_path_image_aug, "horizontal")
                    self.flip_image(full_path_image, full_path_image_aug, "vertical")

            if second_phase:
                # Second phase, place pills on the augmented images
                for _, image_file in tqdm(enumerate(images_files_aug), total=len(images_files_aug)):
                    full_path_image_aug = os.path.join(class_dir_aug, image_file)
                    pill_image_file = random.choice(os.listdir(self.medicine_images))
                    pill_image_path = os.path.join(self.medicine_images, pill_image_file)
                    pill_mask_path = os.path.join(self.medicine_mask, pill_image_file)
                    self.place_medicine_on_tray(pill_image_path, pill_mask_path, full_path_image_aug)

            if third_phase:
                for _, image_file in tqdm(enumerate(images_files_aug_tray), total=len(images_files_aug_tray)):
                    full_path_image_aug = os.path.join(class_image_on_tray, image_file)
                    full_path_tray_aug = os.path.join(class_dir_tray_aug, image_file)

                    self.change_white_balance(full_path_image_aug, full_path_tray_aug, (0.9, 1.0))
                    self.change_brightness(full_path_image_aug, full_path_tray_aug,
                                           exposure_factor=random.uniform(0.5, 1.5))


if __name__ == "__main__":
    aug = AugmentTrayImages()
    aug.main(create_class_dirs=False, first_phase=False, second_phase=False, third_phase=True)
