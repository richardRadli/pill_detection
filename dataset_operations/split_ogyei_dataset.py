import os
import shutil

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector


def filter_filenames_by_class(filenames, class_id):
    return [filename for filename in filenames if int(filename.split("_")[0]) == class_id]


def split_dataset(dataset_path, test_ratio=0.20, random_seed=42):
    image_filenames = [filename for filename in os.listdir(dataset_path) if filename.endswith(".jpg")]
    class_ids = set([int(filename.split("_")[0]) for filename in image_filenames])

    # Select 20% of the classes for the test set
    train_classes, test_classes = train_test_split(list(class_ids),
                                                   test_size=test_ratio,
                                                   random_state=random_seed)
    test_set = [filename for filename in image_filenames if int(filename.split("_")[0]) in test_classes]

    # Calculate the remaining classes after selecting the test set
    remaining_classes = class_ids - set(test_classes)
    train_set, val_set = [], []

    # Split the remaining classes into train and valid sets
    for class_id in remaining_classes:
        class_images = [filename for filename in image_filenames if int(filename.split("_")[0]) == class_id]
        train_image, valid_image = train_test_split(class_images,
                                                    test_size=0.2,
                                                    random_state=random_seed)
        train_set.append(train_image)
        val_set.append(valid_image)

    train_set = [item for sublist in train_set for item in sublist]
    val_set = [item for sublist in val_set for item in sublist]

    return train_set, val_set, test_set, len(image_filenames)


def copy_files(split_set, dataset_images_path, dataset_mask_path, dataset_segmentation_path, dst_dataset_images_path,
               dst_dataset_mask_path, dst_segmentation_path):
    for file in tqdm(split_set):
        src_img = os.path.join(dataset_images_path, file)
        src_mask = os.path.join(dataset_mask_path, file)
        file_txt = file.replace(".jpg", ".txt")
        src_seg = os.path.join(dataset_segmentation_path, file_txt)
        shutil.copy(str(src_img), dst_dataset_images_path)
        shutil.copy(str(src_mask), dst_dataset_mask_path)
        shutil.copy(str(src_seg), dst_segmentation_path)


def set_len(set_name):
    return len(set([int(filename.split('_')[0]) for filename in set_name]))


def main():
    cfg = ConfigAugmentation().parse()

    dataset_images_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("customer").get("customer_images")
    )
    dataset_mask_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("customer").get("customer_mask_images")
    )
    dataset_segmentation_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("customer").get("customer_segmentation_labels")
    )

    dst_train_dataset_images_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("train").get("images")
    )
    dst_train_dataset_mask_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("train").get("mask_images")
    )
    dst_train_segmentation_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("train").get("segmentation_labels")
    )

    dst_valid_dataset_images_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("valid").get("images")
    )
    dst_valid_dataset_mask_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("valid").get("mask_images")
    )
    dst_valid_segmentation_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("valid").get("segmentation_labels")
    )

    dst_test_dataset_images_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("test").get("images")
    )
    dst_test_dataset_mask_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("test").get("mask_images")
    )
    dst_test_segmentation_path = (
        dataset_images_path_selector(dataset_name=cfg.dataset_name).get("test").get("segmentation_labels")
    )

    train_set, val_set, test_set, number_of_images = split_dataset(dataset_path=dataset_images_path)

    # Print the number of classes and instances in each set
    print(f"Number of classes in training set: {set_len(train_set)}")
    print(f"Number of classes in validation set: {set_len(val_set)}")
    print(f"Number of classes in testing set: {set_len(test_set)}")

    # Print the number of instances in each set
    print(f"Number of instances in training set: {len(train_set)}")
    print(f"Number of instances in validation set: {len(val_set)}")
    print(f"Number of instances in testing set: {len(test_set)}")

    print(f"Split ratio, train: {(len(train_set) / number_of_images) * 100:.4f}%, "
          f"valid: {(len(val_set) / number_of_images) * 100:.4f}%, "
          f"test: {(len(test_set) / number_of_images) * 100:.4f}%")

    copy_files(
        split_set=train_set,
        dataset_images_path=dataset_images_path,
        dataset_mask_path=dataset_mask_path,
        dataset_segmentation_path=dataset_segmentation_path,
        dst_dataset_images_path=dst_train_dataset_images_path,
        dst_dataset_mask_path=dst_train_dataset_mask_path,
        dst_segmentation_path=dst_train_segmentation_path
    )

    copy_files(
        split_set=val_set,
        dataset_images_path=dataset_images_path,
        dataset_mask_path=dataset_mask_path,
        dataset_segmentation_path=dataset_segmentation_path,
        dst_dataset_images_path=dst_valid_dataset_images_path,
        dst_dataset_mask_path=dst_valid_dataset_mask_path,
        dst_segmentation_path=dst_valid_segmentation_path
    )

    copy_files(
        split_set=test_set,
        dataset_images_path=dataset_images_path,
        dataset_mask_path=dataset_mask_path,
        dataset_segmentation_path=dataset_segmentation_path,
        dst_dataset_images_path=dst_test_dataset_images_path,
        dst_dataset_mask_path=dst_test_dataset_mask_path,
        dst_segmentation_path=dst_test_segmentation_path
    )


if __name__ == "__main__":
    main()
