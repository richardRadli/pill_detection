import os
import random
import shutil


def create_subdirectories(base_dir):
    subdirs = ['bbox_labels', 'labels', 'images']
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)


def copy_files(root_directory, files, operation_directory):
    for file in files:
        src_image = os.path.join(root_directory, 'images', file)
        dst_image = os.path.join(operation_directory, 'images', file)
        shutil.copy(str(src_image), str(dst_image))

        src_bbox = os.path.join(root_directory, 'bbox_labels', file.replace('.jpg', '.txt'))
        dst_bbox = os.path.join(operation_directory, 'bbox_labels', file.replace('.jpg', '.txt'))
        shutil.copy(str(src_bbox), str(dst_bbox))

        src_label = os.path.join(root_directory, 'labels', file.replace('.jpg', '.txt'))
        dst_label = os.path.join(operation_directory, 'labels', file.replace('.jpg', '.txt'))
        shutil.copy(str(src_label), str(dst_label))


def split_images(root_directory, train_directory, valid_directory, test_directory, sr):
    create_subdirectories(train_directory)
    create_subdirectories(valid_directory)
    create_subdirectories(test_directory)

    image_files = [f for f in os.listdir(os.path.join(root_directory, 'images')) if f.endswith('.jpg')]

    class_files = {}
    for file in image_files:
        class_label = file.split('_')[0]
        if class_label not in class_files:
            class_files[class_label] = []
        class_files[class_label].append(file)

    for class_label in class_files:
        random.shuffle(class_files[class_label])

    train_files = []
    valid_files = []
    test_files = []

    for class_label in class_files:
        split_index1 = int(len(class_files[class_label]) * sr[0])
        split_index2 = int(len(class_files[class_label]) * (sr[0] + sr[1]))
        train_files.extend(class_files[class_label][:split_index1])
        valid_files.extend(class_files[class_label][split_index1:split_index2])
        test_files.extend(class_files[class_label][split_index2:])

    copy_files(root_directory, train_files, train_directory)
    copy_files(root_directory, valid_files, valid_directory)
    copy_files(root_directory, test_files, test_directory)


if __name__ == '__main__':
    split_ratio = [0.64, 0.16, 0.2]

    root_dir = "D:/storage/IVM/datasets/cure/Customer"
    train_dir = "D:/storage/IVM/datasets/cure/train_dir"
    valid_dir = "D:/storage/IVM/datasets/cure/valid_dir"
    test_dir = "D:/storage/IVM/datasets/cure/test_dir"

    split_images(root_dir, train_dir, valid_dir, test_dir, split_ratio)
