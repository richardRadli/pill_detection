import os
import random
import shutil
from tqdm import tqdm

from config.dataset_paths_selector import dataset_images_path_selector
from config.json_config import json_config_selector
from utils.utils import load_config_json


def copy_images(dataset, src_subset, dst_subset, stream):
    random.seed(42)

    src_path = dataset_images_path_selector(dataset).get("src_stream_images").get(src_subset).get(f"stream_images_{stream}")
    dst_path = dataset_images_path_selector(dataset).get("dst_stream_images").get(dst_subset)
    dst_path = str(os.path.join(dst_path, stream))

    pill_classes = []
    for root, dirs, files in os.walk(src_path, topdown=False):
        for name in dirs:
            if name not in pill_classes:
                pill_classes.append(name)

    test_size = int(0.2 * len(pill_classes))
    test_pill_classes = random.sample(pill_classes, test_size)
    test_pill_classes_set = set(test_pill_classes)

    pill_classes_set = set(pill_classes)
    train_pill_classes_set = pill_classes_set - test_pill_classes_set
    train_pill_classes = sorted(list(train_pill_classes_set))

    pill_class = train_pill_classes if dst_subset in ["stream_images_anchor", "stream_images_pos_neg"] else test_pill_classes

    for pill_class in tqdm(pill_class, total=len(pill_class), desc="Copying images"):
        src_class_path = str(os.path.join(src_path, pill_class))
        dst_class_path = os.path.join(dst_path, pill_class)

        os.makedirs(dst_class_path, exist_ok=True)

        for file_name in os.listdir(src_class_path):
            src_file = os.path.join(src_class_path, file_name)
            dst_file = os.path.join(dst_class_path, file_name)

            shutil.copy(src_file, dst_file)


def main():
    cfg = load_config_json(
        json_filename=json_config_selector("stream_images").get("config"),
        json_schema_filename=json_config_selector("stream_images").get("schema"),
    )

    src_subsets = ["customer", "reference"]
    dst_subsets = {
        "customer": ["stream_images_pos_neg", "query"],
        "reference": ["stream_images_anchor", "reference"]
    }
    substreams = ["contour", "lbp", "rgb", "texture"]

    for src_subset in src_subsets:
        for dst_subset in dst_subsets[src_subset]:
            for substream in substreams:
                copy_images(cfg["dataset_type"], src_subset, dst_subset, substream)


if __name__ == "__main__":
    main()
