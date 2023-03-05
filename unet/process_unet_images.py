import cv2
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

from const import CONST
from utils.dataset_operations import create_label_dirs, numerical_sort


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ D R A W   B O U N D I N G   B O X -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def draw_bounding_box_images() -> None:
    """
    Reads in the images, draws the bounding box, and saves the images.

    :return: None
    """

    # Make sure the output directory exists
    os.makedirs(CONST.dir_bounding_box, exist_ok=True)

    # Read in all color and mask images at once
    color_images = cv2.imreadmulti(sorted(glob(CONST.dir_test_images + "/*.png")), flags=cv2.IMREAD_COLOR)[1]
    mask_images = cv2.imreadmulti(sorted(glob(CONST.dir_unet_output + "/*.png")), flags=cv2.IMREAD_GRAYSCALE)[1]

    for idx, (c_imgs, m_imgs) in tqdm(enumerate(zip(color_images, mask_images)), total=len(color_images),
                                      desc="Bounding box"):
        output_name = os.path.basename(color_images[idx])
        output_file = os.path.join(CONST.dir_bounding_box, output_name)

        # Find the largest bounding box using minMaxLoc
        _, max_val, _, max_loc = cv2.minMaxLoc(m_imgs)
        if max_val > 0:
            x, y, w, h = cv2.connectedComponentsWithStats((m_imgs == max_val).astype(np.uint8), connectivity=8,
                                                          ltype=cv2.CV_32S)[2][1]
            # Crop the object and save to file
            obj = c_imgs[y:y + h, x:x + w]
            cv2.imwrite(output_file, obj)


def create_contour_images(cropped_image, output_path: str, kernel_size: tuple = (5, 5), canny_low_thr: int = 15,
                          canny_high_thr: int = 40) -> None:
    """
    Applies Gaussian blur on the images to smooth the edges, after the function carries out Canny edge detection.
    Finally, it saves the images to a given directory.

    param cropped_image: current image to process
    param output_path: path where the processed image should be saved
    param kernel_size: Size of the kernel in the Gaussian blurring function
    param canny_low_thr: lower threshold limit
    param canny_high_thr: upper threshold limit
    :return: None
    """

    blured_images = cv2.GaussianBlur(cropped_image, kernel_size, 0)
    edges = cv2.Canny(blured_images, canny_low_thr, canny_high_thr)
    cv2.imwrite(output_path, edges)


def save_contour_images() -> None:
    """
    Executes the create_contour_images() function in a multithread way.

    :return: None
    """

    contour_images = sorted(glob(CONST.dir_bounding_box + "/*.png"), key=numerical_sort)
    with ThreadPoolExecutor() as executor:
        futures = []
        for img_path in contour_images:
            output_name = "contour_" + img_path.split("\\")[2]
            output_file = (os.path.join(CONST.dir_contour, output_name))
            bbox_imgs = cv2.imread(img_path, 0)
            futures.append(executor.submit(create_contour_images, bbox_imgs, output_file))


def create_texture_image(img_path: str, output_path: str, kernel_size: tuple = (7, 7)) -> None:
    """
    Creates texture image for a given input image

    :param img_path: Input path of the image to be processed.
    :param output_path: Output path of the processed image.
    :param kernel_size: Size of the kernel of the Gaussian Blur function.
    :return: None
    """

    img = cv2.imread(img_path, 0)
    blured_images = cv2.GaussianBlur(img, kernel_size, 0)
    sub_img = img - blured_images
    cv2.imwrite(output_path, sub_img)


def save_texture_images() -> None:
    """
    Executes the create_texture_image() function in a multithread way. Saves texture images.

    :return: None
    """

    bbox_images = sorted(glob(CONST.dir_bounding_box + "/*.png"), key=numerical_sort)
    output_paths = [(os.path.join(CONST.dir_texture, "texture_" + img_path.split("\\")[2]), img_path) for img_path in
                    bbox_images]

    with Pool() as pool:
        list(tqdm(pool.imap_unordered(create_texture_image, output_paths), total=len(output_paths),
                  desc="Texture images"))


def main():
    """

    :return:
    """

    draw_bounding_box_images()
    # save_contour_images()
    # save_texture_images()
    # create_label_dirs(CONST.dir_bounding_box)
    # create_label_dirs(CONST.dir_contour)
    # create_label_dirs(CONST.dir_texture)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as kie:
        print(kie)
