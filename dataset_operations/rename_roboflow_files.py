import os
from glob import glob
from PIL import Image


def rename_files():
    path = "C:/Users/ricsi/Desktop/test"
    images = sorted(glob(path + "/images/*.jpg"))
    text = sorted(glob(path + "/labels/*.txt"))

    existing_files = os.listdir(path + "/images")

    for idx, (img, txt) in enumerate(zip(images, text)):
        image_file_name = os.path.basename(img)
        image_file_name = image_file_name.replace("_png", ".png")
        image_file_name = '_'.join(image_file_name.split('.')[:2])
        image_file_name = image_file_name.replace("_png", ".png")

        txt_file_name = os.path.basename(txt)
        txt_file_name = txt_file_name.replace("_png", ".png")
        txt_file_name = '_'.join(txt_file_name.split('.')[:2])
        txt_file_name = txt_file_name.replace("_png", ".txt")

        # Check if the image file name already exists
        original_file_name = image_file_name
        index = 0
        while image_file_name in existing_files:
            index += 1
            image_file_name = "{}_{:03d}.png".format(original_file_name.rsplit('_', 1)[0], index)

        existing_files.append(image_file_name)

        # Rename the corresponding text file
        os.rename(img, os.path.join(path, "images", image_file_name))
        os.rename(txt, os.path.join(path, "labels", txt_file_name))

        # Print the updated file names
        print("Image file:", image_file_name)
        print("Text file:", txt_file_name)


def convert_images_to_png(directory):
    image_files = os.listdir(directory)
    for file_name in image_files:
        if file_name.lower().endswith(('.png')):
            image_path = os.path.join(directory, file_name)
            img = Image.open(image_path)
            img = img.convert("RGB")

            img.save(image_path, 'PNG')
            print(f"Converted {file_name}")


convert_images_to_png("C:/Users/ricsi/Desktop/test/images")