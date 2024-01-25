import os


def modify_class_id(file_name, dst_path):
    try:
        # file_name = os.path.basename(file_name)
        # class_id = int(file_name.split('_')[0])
        class_id = 0

        with open(file_name, 'r') as file:
            content = file.read()

        words = content.split()
        modified_content = content.replace(words[0], str(class_id), 1)

        dst_filename = os.path.join(dst_path, os.path.basename(file_name))
        with open(dst_filename, 'w') as file:
            file.write(modified_content)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    directory_path = f"D:/storage/IVM/datasets/cure/Customer_augmented/valid_dir/yolo_labels/"
    dest_path = f"D:/storage/IVM/datasets/cure/Customer_augmented/valid_dir/yolo_binary_labels/"

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            modify_class_id(file_path, dest_path)
