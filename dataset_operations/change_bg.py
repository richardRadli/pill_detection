import cv2
import os
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from glob import glob
from tqdm import tqdm


def change_background_dtd(image_path: str, mask_path: str, output_dir) -> None:
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

    assert image.shape[:2] == mask.shape[:2]

    color = (145, 145, 151)
    background = np.ones(image.shape, dtype=np.uint8)
    background[:, :, 0] = color[0]
    background[:, :, 1] = color[1]
    background[:, :, 2] = color[2]

    foreground = cv2.bitwise_and(image, image, mask=mask)
    background = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))

    output_image = cv2.add(foreground, background)
    file_name = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(file_name, output_image)


if __name__ == "__main__":
    images_path = sorted(glob("D:/storage/pill_detection/datasets/ogyei/Reference/images_original/*"))
    mask_path = sorted(glob("D:/storage/pill_detection/datasets/ogyei/Reference/mask_images/*"))
    output_path = "D:/storage/pill_detection/datasets/ogyei/Reference/images"

    with ThreadPoolExecutor() as executor:
        futures = []
        for img, mask in zip(images_path, mask_path):
            assert os.path.basename(img) == os.path.basename(mask)
            future = executor.submit(change_background_dtd, img, mask, output_path)
            futures.append(future)

        # Wait for all futures to complete
        for future in tqdm(futures, total=len(images_path)):
            future.result()

