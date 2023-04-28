import concurrent.futures
import cv2
import os

from glob import glob
from tqdm import tqdm


def convert_yolo_format_to_pixels(image, annotation):
    # Get the dimensions of the cropped image
    cropped_height, cropped_width, _ = image.shape

    # Convert the annotation coordinates to pixel values
    annotation_points = []
    if len(annotation) % 2 == 0:
        for i in range(0, len(annotation), 2):
            x = int(annotation[i] * cropped_width)
            y = int(annotation[i + 1] * cropped_height)
            annotation_points.append((x, y))

    return annotation_points


def convert_coordinates_to_original(cropped_coordinates, cropped_width, cropped_height, original_width,
                                    original_height):
    cropped_x, cropped_y = cropped_coordinates

    # Calculate the center position of the cropped image
    cropped_center_x = original_width / 2
    cropped_center_y = original_height / 2

    # Calculate the offset from the center due to cropping
    offset_x = cropped_center_x - cropped_width / 2
    offset_y = cropped_center_y - cropped_height / 2

    # Calculate the corresponding positions on the original image
    original_x = int(cropped_x + offset_x)
    original_y = int(cropped_y + offset_y)

    return original_x, original_y


def convert_pixels_to_yolo_format(image_width, image_height, coordinates, class_id):
    yolo_coordinates = [class_id]
    for x, y in coordinates:
        x_normalized = x / image_width
        y_normalized = y / image_height
        yolo_coordinates.extend([x_normalized, y_normalized])
    return yolo_coordinates


def read_image_to_list(dir_train_images):
    subdirs = sorted(glob(os.path.join(dir_train_images, "*", "")))
    file_names = []

    for subdir in subdirs:
        images = sorted(glob(os.path.join(subdir, "*.png")))
        for idx, img_path in tqdm(enumerate(images), total=len(images)):
            file_names.append(img_path)

    return file_names


def read_yolo_annotations_to_list(yolo_dir):
    txt_files = sorted(glob(os.path.join(yolo_dir, "*.txt")))
    file_names = []

    for idx, txt_file in tqdm(enumerate(txt_files), total=len(txt_files)):
        file_names.append(txt_file)

    return file_names


def save_text_to_file(data_list, file_path):
    with open(file_path, "w") as file:
        for item in data_list:
            file.write(str(item) + "\n")


def process_image(main_dir, ori, cropped, yolo_annotation):
    cropped_img = cv2.imread(cropped)
    original_img = cv2.imread(ori)
    with open(yolo_annotation, "r") as file:
        annotation_text = file.readline().strip()

    annotation_list = list(map(float, annotation_text.split()))
    class_id = int(annotation_list[0])
    annotation_list = annotation_list[1:]
    original_pixel_coordinates = []

    annotation_points = convert_yolo_format_to_pixels(image=cropped_img, annotation=annotation_list)
    for c in annotation_points:
        original_x, original_y = convert_coordinates_to_original(c, cropped_img.shape[1], cropped_img.shape[0],
                                                                 original_img.shape[1], original_img.shape[0])
        original_pixel_coordinates.append((original_x, original_y))

    yolo_coordinates = convert_pixels_to_yolo_format(original_img.shape[1], original_img.shape[0],
                                                     original_pixel_coordinates, class_id)

    save_text_to_file(data_list=yolo_coordinates,
                      file_path=os.path.join(main_dir, "train_labels_ori", os.path.basename(yolo_annotation)))


def main():
    main_dir = "D:/project/IVM"
    original_imgs_file_names = \
        read_image_to_list(os.path.join(main_dir, "captured_OGYEI_pill_photos_undistorted"))
    cropped_imgs_file_names = \
        read_image_to_list(os.path.join(main_dir, "captured_OGYEI_pill_photos_undistorted_cropped"))
    yolo_annotations = read_yolo_annotations_to_list(os.path.join(main_dir, "train_labels"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_image, [main_dir]*len(original_imgs_file_names),
                     original_imgs_file_names, cropped_imgs_file_names, yolo_annotations)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as kie:
        print(kie)
