import os
import random
import shutil

from config.dataset_paths_selector import dataset_images_path_selector

random.seed(42)

src_path = dataset_images_path_selector("ogyei").get("src_stream_images").get("reference").get("stream_images_contour")
dst_path = dataset_images_path_selector("ogyei").get("dst_stream_images").get("ref")
dst_path = os.path.join(dst_path, "contour")

pill_classes = []
for root, dirs, files in os.walk(src_path, topdown=False):
    for name in dirs:
        if name not in pill_classes:
            pill_classes.append(name)

pill_classes_set = set(pill_classes)

test_size = int(0.2 * len(pill_classes))
test_pill_classes = random.sample(pill_classes, test_size)
test_pill_classes_set = set(test_pill_classes)

train_pill_classes_set = pill_classes_set - test_pill_classes_set
train_pill_classes = sorted(list(train_pill_classes_set))

for pill_class in test_pill_classes:
    src_class_path = os.path.join(src_path, pill_class)
    dst_class_path = os.path.join(dst_path, pill_class)

    os.makedirs(dst_class_path, exist_ok=True)

    for file_name in os.listdir(src_class_path):
        src_file = os.path.join(src_class_path, file_name)
        dst_file = os.path.join(dst_class_path, file_name)

        shutil.copy(src_file, dst_file)
