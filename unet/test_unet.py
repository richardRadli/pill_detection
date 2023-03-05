import os
import glob

from tqdm import tqdm

from const import CONST
from utils.utils import find_latest_file


def predict_multiple_images():
    png_files = glob.glob(CONST.dir_test_images + "/*.png")
    latest_model = find_latest_file(CONST.dir_checkpoint)

    for name in tqdm(png_files):
        file_name = name[:-4] + "_OUT.png"
        out_path = os.path.join(CONST.dir_unet_output, file_name.split("\\")[2])
        os.system("python predict_unet.py "
                  f"-i {name} "
                  f"-o {out_path} "
                  f"-m {latest_model}"
                  )


def predict_single_image():
    os.system("python predict_unet.py "
              "-i C:/Users/keplab/Desktop/clahe.png "
              "-o C:/Users/keplab/Desktop/out.png "
              "-m D:/project/IVM/data/checkpoints/2023-02-12_04-47-10/checkpoint_epoch200.pth "
              )


# predict_single_image()
