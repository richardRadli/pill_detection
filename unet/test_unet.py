import os
import glob

from tqdm import tqdm

from const import CONST
from utils.utils import find_latest_file


def predict_multiple_images():
    png_files = glob.glob(CONST.dir_test_images + "/*.png")
    latest_model = find_latest_file(CONST.dir_unet_checkpoint)

    for name in tqdm(png_files):
        file_name = name[:-4] + "_OUT.png"
        out_path = os.path.join(CONST.dir_unet_output, file_name.split("\\")[2])
        os.system("python predict_unet.py "
                  f"-i {name} "
                  f"-o {out_path} "
                  f"-m {latest_model} "
                  )


def predict_single_image():
    os.system("python predict_unet.py "
              "-i C:/Users/ricsi/Desktop/in_osszeer.png "
              "-o C:/Users/ricsi/Desktop/out_osszeer.png "
              "-m C:/Users/ricsi/Documents/project/storage/IVM/data/unet_checkpoints/"
              "2023-04-14_15-09-16/checkpoint_epoch5.pth "
              )


if __name__ == "__main__":
    predict_single_image()
