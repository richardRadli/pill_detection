import cv2
import os

from glob import glob
from tqdm import tqdm

from const import CONST
from utils.dataset_operations import create_label_dirs, numerical_sort


def draw_bounding_box(in_img, seg_map, output_path: str):
    """
    Draws bounding box over medicines. It draws only the biggest bounding box, small ones are terminated. After that it
    crops out the bounding box's content.
    param in_img: input testing image
    param seg_map: output of the unet for the input testing image
    param output_path: where the file should be saved
    :return:None
    """

    ret, thresh = cv2.threshold(seg_map, 0, 255, cv2.THRESH_BINARY)
    n_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=cv2.CV_32S)

    for i in range(1, n_objects):
        x, y, w, h, area = stats[i]
        if area > 20000:
            obj = in_img[y:y + h, x:x + w]
            cv2.imwrite(output_path, obj)


def save_bounding_box_images():
    """
    Reads in the images, draws the bounding box, and saves the images.
    :return: None
    """

    color_images = sorted(glob(CONST.dir_test_images + "/*.png"))
    mask_images = sorted(glob(CONST.dir_unet_output + "/*.png"))

    for idx, (color_imgs, mask_imgs) in tqdm(enumerate(zip(color_images, mask_images)), total=len(color_images),
                                             desc="Bounding box"):
        output_name = color_imgs.split("\\")[2]
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
    for _, img_path in tqdm(enumerate(contour_images), total=len(contour_images), desc="Contour images"):
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
    for _, img_path in tqdm(enumerate(bbox_images), total=len(bbox_images), desc="Texture images"):
        output_name = "texture_" + img_path.split("\\")[2]
        output_file = (os.path.join(CONST.dir_texture, output_name))
        texture_images = cv2.imread(img_path, 0)
        create_texture_images(texture_images, output_file)


def main():
    """

    :return:
    """

    save_bounding_box_images()
    save_contour_images()
    save_texture_images()
    create_label_dirs(CONST.dir_bounding_box)
    create_label_dirs(CONST.dir_contour)
    create_label_dirs(CONST.dir_texture)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as kie:
        print(kie)
