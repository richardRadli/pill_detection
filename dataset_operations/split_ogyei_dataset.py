import os
import shutil

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------- F I L T E R   F I L E N A M E S   B Y   C L A S S E S --------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def filter_filenames_by_class(filenames: list[str], class_id: int) -> list[str]:
    """
    Filter filenames based on class_id.

    Args:
        filenames (list[str]): List of filenames.
        class_id (int): Class ID.

    Returns:
        list[str]: Filtered list of filenames.
    """

    return [filename for filename in filenames if int(filename.split("_")[0]) == class_id]


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- S P L I T   D A T A S E T ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def split_dataset(dataset_path: str, test_ratio: float = 0.20, random_seed: int = 42) ->\
        tuple[list[str], list[str], list[str], int]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        dataset_path (str): Path to the dataset.
        test_ratio (float): Ratio of the dataset to be used for testing.
        random_seed (int): Random seed for reproducibility.

    Returns:
        tuple[list[str], list[str], list[str], int]: Train set, validation set, test set, total number of images.
    """

    image_filenames = [filename for filename in os.listdir(dataset_path) if filename.endswith(".jpg")]
    class_ids = set([int(filename.split("_")[0]) for filename in image_filenames])

    # Select test classes
    train_classes, test_classes = train_test_split(list(class_ids),
                                                   test_size=test_ratio,
                                                   random_state=random_seed)
    test_set = [filename for filename in image_filenames if int(filename.split("_")[0]) in test_classes]

    # Select train and validation classes
    remaining_classes = class_ids - set(test_classes)
    train_set, val_set = [], []

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


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- C O P Y   F I L E S ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def copy_files(split_set: list[str], dataset_images_path: str, dataset_mask_path: str, dataset_segmentation_path: str,
               dst_dataset_images_path: str, dst_dataset_mask_path: str, dst_segmentation_path: str) -> None:
    """
    Copy files from source to destination directories.

    Args:
        split_set (list[str]): List of files to copy.
        dataset_images_path (str): Path to dataset images.
        dataset_mask_path (str): Path to dataset masks.
        dataset_segmentation_path (str): Path to dataset segmentations.
        dst_dataset_images_path (str): Destination path for images.
        dst_dataset_mask_path (str): Destination path for masks.
        dst_segmentation_path (str): Destination path for segmentations.

    Returns:
        None
    """

    for file in tqdm(split_set):
        src_img = os.path.join(dataset_images_path, file)
        src_mask = os.path.join(dataset_mask_path, file)
        file_txt = file.replace(".jpg", ".txt")
        src_seg = os.path.join(dataset_segmentation_path, file_txt)
        shutil.copy(str(src_img), dst_dataset_images_path)
        shutil.copy(str(src_mask), dst_dataset_mask_path)
        shutil.copy(str(src_seg), dst_segmentation_path)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- S E T   L E N ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def set_len(set_name: list[str]) -> int:
    """
    Get the length of the set.

    Args:
        set_name (list[str]): Set of filenames.

    Returns:
        int: Length of the set.
    """

    return len(set([int(filename.split('_')[0]) for filename in set_name]))


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- M A I N ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    """
    Executes the functions above.

    Returns:
        None
    """

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
