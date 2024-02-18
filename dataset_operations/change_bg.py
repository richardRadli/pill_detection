import cv2
import os
import numpy as np


def change_background_dtd(image_path: str, mask_path: str, output_dir) -> None:
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

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
    from glob import glob
    images_path = sorted(glob("D:/storage/pill_detection/datasets/cure/Customer/images_original/*"))
    mask_path = sorted(glob("D:/storage/pill_detection/datasets/cure/Customer/mask_images/*"))
    output_path = "D:/storage/pill_detection/datasets/cure/Customer/images"

    for img, mask in zip(images_path, mask_path):
        change_background_dtd(
            image_path=img,
            mask_path=mask,
            output_dir=output_path
    )
