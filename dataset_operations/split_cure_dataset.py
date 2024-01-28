import os
import shutil

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector


def filter_filenames_by_class(filenames, class_id):
    return [filename for filename in filenames if int(filename.split("_")[0]) == class_id]


def split_dataset(dataset_path, val_ratio=0.16, test_ratio=0.20, random_seed=42):
    image_filenames = [filename for filename in os.listdir(dataset_path) if filename.endswith(".jpg")]
    class_ids = set([int(filename.split("_")[0]) for filename in image_filenames])

    train_classes, test_classes = train_test_split(list(class_ids),
                                                   test_size=test_ratio,
                                                   random_state=random_seed)
    train_classes, val_classes = train_test_split(train_classes,
                                                  test_size=val_ratio / (1 - test_ratio),
                                                  random_state=random_seed)

    train_set = \
        [filename for class_id in train_classes for filename in filter_filenames_by_class(image_filenames, class_id)]
    val_set = \
        [filename for class_id in val_classes for filename in filter_filenames_by_class(image_filenames, class_id)]
    test_set = \
        [filename for class_id in test_classes for filename in filter_filenames_by_class(image_filenames, class_id)]

    return train_set, val_set, test_set


def copy_files(split_set, dataset_images_path, dataset_bbox_path, dataset_segmentation_path, dst_dataset_images_path,
               dst_dataset_bbox_path, dst_segmentation_path):
    for file in tqdm(split_set):
        src_img = os.path.join(dataset_images_path, file)
        file_txt = file.replace(".jpg", ".txt")
        src_bbox = os.path.join(dataset_bbox_path, file_txt)
        src_seg = os.path.join(dataset_segmentation_path, file_txt)
        shutil.copy(str(src_img), dst_dataset_images_path)
        shutil.copy(str(src_bbox), dst_dataset_bbox_path)
        shutil.copy(str(src_seg), dst_segmentation_path)


def main():
    cfg = ConfigAugmentation().parse()

    dataset_images_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("customer_images")
    )
    dataset_bbox_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("customer_pixel_bbox_labels")
    )
    dataset_segmentation_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("customer_segmentation_labels")
    )

    dst_train_dataset_images_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("train_images")
    )
    dst_train_dataset_bbox_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("train_bbox_pixel_labels")
    )
    dst_train_segmentation_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("train_segmentation_labels")
    )

    dst_valid_dataset_images_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("valid_images")
    )
    dst_valid_dataset_bbox_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("valid_bbox_pixel_labels")
    )
    dst_valid_segmentation_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("valid_segmentation_labels")
    )

    dst_test_dataset_images_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("test_images")
    )
    dst_test_dataset_bbox_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("test_bbox_pixel_labels")
    )
    dst_test_segmentation_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("test_segmentation_labels")
    )

    train_set, val_set, test_set = split_dataset(dataset_images_path)

    # Print the number of classes and instances in each set
    print(f"Number of classes in training set: {len(set([int(filename.split('_')[0]) for filename in train_set]))}")
    print(f"Number of classes in validation set: {len(set([int(filename.split('_')[0]) for filename in val_set]))}")
    print(f"Number of classes in testing set: {len(set([int(filename.split('_')[0]) for filename in test_set]))}")

    # Print the number of instances in each set
    print(f"Number of instances in training set: {len(train_set)}")
    print(f"Number of instances in validation set: {len(val_set)}")
    print(f"Number of instances in testing set: {len(test_set)}")

    copy_files(
        split_set=train_set,
        dataset_images_path=dataset_images_path,
        dataset_bbox_path=dataset_bbox_path,
        dataset_segmentation_path=dataset_segmentation_path,
        dst_dataset_images_path=dst_train_dataset_images_path,
        dst_dataset_bbox_path=dst_train_dataset_bbox_path,
        dst_segmentation_path=dst_train_segmentation_path
    )

    copy_files(
        split_set=val_set,
        dataset_images_path=dataset_images_path,
        dataset_bbox_path=dataset_bbox_path,
        dataset_segmentation_path=dataset_segmentation_path,
        dst_dataset_images_path=dst_valid_dataset_images_path,
        dst_dataset_bbox_path=dst_valid_dataset_bbox_path,
        dst_segmentation_path=dst_valid_segmentation_path
    )

    copy_files(
        split_set=test_set,
        dataset_images_path=dataset_images_path,
        dataset_bbox_path=dataset_bbox_path,
        dataset_segmentation_path=dataset_segmentation_path,
        dst_dataset_images_path=dst_test_dataset_images_path,
        dst_dataset_bbox_path=dst_test_dataset_bbox_path,
        dst_segmentation_path=dst_test_segmentation_path
    )


if __name__ == "__main__":
    main()
