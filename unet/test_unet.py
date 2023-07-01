import os
import glob

from tqdm import tqdm

from config.const import DATA_PATH, DATASET_PATH, IMAGES_PATH
from utils.utils import find_latest_file_in_latest_directory


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------- P R E D I C T   M U L T I P L E   I M A G E S ------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def predict_multiple_images(input_dir: str, output_dir: str) -> None:
    """

    :param input_dir: path to the directory, that holds the test images
    :param output_dir: path to the directory, where the images will be saved
    :return: None
    """

    png_files = glob.glob(input_dir + "/*.png")
    latest_model = find_latest_file_in_latest_directory(DATA_PATH.get_data_path("weights_unet"))

    for name in tqdm(png_files):
        file_name = name[:-4] + "_OUT.png"
        out_path = os.path.join(output_dir, os.path.basename(file_name))
        os.system("python predict_unet.py "
                  f"-i {name} "
                  f"-o {out_path} "
                  f"-m {latest_model} ")


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- P R E D I C T   S I N G L E   I M A G E S --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def predict_single_image(input_image_file_path: str, output_image_file_path: str) -> None:
    """

    :param input_image_file_path: path to the input file
    :param output_image_file_path: path to where the image will be saved
    :return: None
    """

    # model_path = find_latest_file_in_latest_directory(DATA_PATH.get_data_path("weights_unet"))
    model_path = "C:/Users/ricsi/Desktop/cure/saves/checkpoint_epoch5.pth"
    os.system("python predict_unet.py "
              f"-i {input_image_file_path} "
              f"-o {output_image_file_path} "
              f"-m {model_path} "
              )


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ M A I N -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main(operation: str = "multi") -> None:
    """
    This function executes the prediction with UNet

    :param operation: either "multi" or "single"
    :return: None
    """

    if operation.lower() == "multi":
        predict_multiple_images(input_dir=DATASET_PATH.get_data_path("ogyi_v2_splitted_test_images"),
                                output_dir=IMAGES_PATH.get_data_path("unet_out"))
    elif operation.lower() == "single":
        predict_single_image(input_image_file_path="C:/Users/ricsi/Desktop/108_bottom_108_ref_bottom_brightness_2_brightness_2.png",
                             output_image_file_path="C:/Users/ricsi/Desktop/out.jpg")
    else:
        raise ValueError("Wrong operation!")


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- __M A I N__ -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main(operation="single")
