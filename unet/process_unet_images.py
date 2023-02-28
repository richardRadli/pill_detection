import cv2
import os
import re

from glob import glob

from const import CONST


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def draw_bounding_box(in_img, seg_map, output_path: str):
    """
    Draws bounding box over medicines. It draws only the biggest bounding box, small ones are terminated. After that it
    crops out the bounding box's content.
    param in_img: input testing image
    param seg_map: output of the unet for the input testing image
    param output_path: where the file should be saved
    :return:None
    """

    # Find the contours in the grayscale image
    contours, hierarchy = cv2.findContours(seg_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Draw a bounding box for each contour
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_img = in_img[y:y + h, x:x + w]

        # Save the segmentation map with bounding boxes
        cv2.imwrite(output_path, cropped_img)


def save_bounding_box_images():
    """
    Reads in the images, draws the bounding box, and saves the images.
    :return: None
    """

    color_images = sorted(glob(CONST.dir_test_images + "/*.png"))
    mask_images = sorted(glob(CONST.dir_unet_output + "/*.png"))

    for idx, (color_imgs, mask_imgs) in enumerate(zip(color_images, mask_images)):
        output_name = "bbox_" + color_imgs.split("\\")[2]
        output_file = (os.path.join(CONST.dir_bounding_box, output_name))
        c_imgs = cv2.imread(color_imgs, 1)
        m_imgs = cv2.imread(mask_imgs, 0)
        draw_bounding_box(c_imgs, m_imgs, output_file)


def create_contour_images(cropped_image, output_path: str, kernel_size: tuple = (5, 5), canny_low_thr: int = 15,
                          canny_high_thr: int = 40):
    """

    param cropped_image:
    param output_path:
    param kernel_size: Size of the kernel in the Gaussian blurring function
    param canny_low_thr:
    param canny_high_thr:
    :return:
    """

    blured_images = cv2.GaussianBlur(cropped_image, kernel_size, 0)
    edges = cv2.Canny(blured_images, canny_low_thr, canny_high_thr)
    cv2.imwrite(output_path, edges)


def save_contour_images():
    """

    :return:
    """

    contour_images = sorted(glob(CONST.dir_bounding_box + "/*.png"), key=numerical_sort)
    for _, img_path in enumerate(contour_images):
        output_name = "contour_" + img_path.split("\\")[2]
        output_file = (os.path.join(CONST.dir_contour, output_name))
        bbox_imgs = cv2.imread(img_path, 0)
        create_contour_images(bbox_imgs, output_file)


def create_texture_images(cropped_image,  output_path, kernel_size: tuple = (7, 7)):
    """

    param cropped_image:
    param output_path:
    param kernel_size:
    :return:
    """

    blured_images = cv2.GaussianBlur(cropped_image, kernel_size, 0)
    sub_img = cropped_image - blured_images
    cv2.imwrite(output_path, sub_img)


def save_texture_images():
    """

    :return:
    """

    bbox_images = sorted(glob(CONST.dir_bounding_box + "/*.png"), key=numerical_sort)
    for _, img_path in enumerate(bbox_images):
        output_name = "texture_" + img_path.split("\\")[2]
        output_file = (os.path.join(CONST.dir_texture, output_name))
        texture_images = cv2.imread(img_path, 0)
        create_texture_images(texture_images, output_file)


def main():
    """

    :return:
    """

    # save_bounding_box_images()
    save_contour_images()
    save_texture_images()


if __name__ == "__main__":
    main()
