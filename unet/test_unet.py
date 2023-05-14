import os
import glob

from tqdm import tqdm

from config.const import CONST
from utils.utils import find_latest_file_in_latest_directory


def predict_multiple_images(input_dir: str, output_dir: str):
    png_files = glob.glob(input_dir + "/*.png")
    latest_model = find_latest_file_in_latest_directory(CONST.dir_unet_checkpoint)

    for name in tqdm(png_files):
        file_name = name[:-4] + "_OUT.png"
        out_path = os.path.join(output_dir, os.path.basename(file_name))
        os.system("python predict_unet.py "
                  f"-i {name} "
                  f"-o {out_path} "
                  f"-m {latest_model} "
                  )


def predict_single_image(input_image_path: str, output_image_path: str):
    model_path = find_latest_file_in_latest_directory(CONST.dir_unet_checkpoint)
    os.system("python predict_unet.py "
              f"-i {input_image_path} "
              f"-o {output_image_path} "
              f"-m {model_path} "
              )


def main(operation: str = "multi"):
    if operation.lower() == "multi":
        predict_multiple_images(input_dir=CONST.dir_test_images,
                                output_dir=CONST.dir_unet_output)
    elif operation.lower() == "single":
        predict_single_image(input_image_path="C:/Users/ricsi/Desktop/id_mul_001_002_003_ambroxol_egis-dorithricin_mentol-cataflam_v_009.png",
                             output_image_path="C:/Users/ricsi/Desktop/out_osszeer.png")
    else:
        raise ValueError("Wrong operation!")


if __name__ == "__main__":
    main(operation="multi")
