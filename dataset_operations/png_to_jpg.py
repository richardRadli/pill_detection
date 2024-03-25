import os

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm


def convert_and_save(src_directory, dst_directory, jpg_file):
    png_path = os.path.join(src_directory, jpg_file)
    jpg_path = os.path.join(dst_directory, os.path.splitext(jpg_file)[0] + '.jpg')

    with Image.open(png_path) as img:
        img = img.convert('RGB')
        img.save(jpg_path)


def convert_png_to_jpg(src_directory, dst_directory, num_threads=4):
    os.makedirs(dst_directory, exist_ok=True)

    png_files = [file for file in os.listdir(src_directory) if file.lower().endswith('.bmp')]

    with (ThreadPoolExecutor(max_workers=num_threads) as executor):
        futures = \
        {executor.submit(convert_and_save, src_directory, dst_directory, jpg_file): jpg_file for jpg_file in png_files}

        for future in tqdm(futures):
            future.result()


if __name__ == "__main__":
    input_directory = "D:/storage/pill_detection/datasets/cure_one_sided/Reference/images"
    output_directory = "D:/storage/pill_detection/datasets/cure_one_sided/Reference/images"
    convert_png_to_jpg(input_directory, output_directory, 6)
    print("Conversion completed.")
