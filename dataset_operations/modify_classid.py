import os


def modify_class_id(filename):
    try:
        file_name = os.path.basename(filename)
        class_id = int(file_name.split('_')[0])

        with open(filename, 'r') as file:
            content = file.read()

        words = content.split()
        modified_content = content.replace(words[0], str(class_id), 1)

        print(file_name, modified_content)
        with open(filename, 'w') as file:
            file.write(modified_content)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    directory_path = f"D:/storage/IVM/datasets/cure/customer_annotated/bbox_labels"

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            modify_class_id(file_path)
