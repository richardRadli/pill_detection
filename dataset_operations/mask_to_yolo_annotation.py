import cv2
import os

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector
from utils.utils import file_reader


def main():
    cfg = ConfigAugmentation().parse()
    annotation_path = dataset_images_path_selector(cfg.dataset_name).get("reference").get("reference_labels")
    masks = dataset_images_path_selector(cfg.dataset_name).get("reference").get("reference_masks")
    masks_path = file_reader(masks, "jpg")

    for idx, masks in enumerate(masks_path):
        basename = os.path.basename(masks)
        class_id = int(basename.split('_')[0])
        mask = cv2.imread(masks, cv2.IMREAD_GRAYSCALE)
        anno_file = os.path.join(annotation_path, os.path.basename(masks))
        anno_file = anno_file.replace(".png", ".txt")

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

        with open(anno_file, 'w') as file:
            for box in bounding_boxes:
                x, y, w, h = box
                x /= mask.shape[1]
                y /= mask.shape[0]
                w /= mask.shape[1]
                h /= mask.shape[0]

                file.write(f'{str(class_id)} {x + w / 2} {y + h / 2} {w} {h}\n')


if __name__ == '__main__':
    main()
