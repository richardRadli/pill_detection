import os
import glob

from tqdm import tqdm

from const import CONST

# Get a list of all PNG files in the directory
png_files = glob.glob(CONST.dir_test_images + "/*.png")

for name in tqdm(png_files):
    file_name = name[:-4] + "_OUT.png"
    out_path = os.path.join(CONST.dir_unet_output, file_name.split("\\")[2])
    os.system("python predict_unet.py "
              f"-i {name} "
              f"-o  {out_path} "
              "-m E:/users/ricsi/IVM/checkpoints/2023.02.20/checkpoint_epoch200.pth"
              )
