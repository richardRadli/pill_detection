import os
import shutil
from const import CONST


def hardest_samples(directory):
    paths = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as f:
                data = eval(f.read())

            for key in data:
                paths.append(data[key])

    return paths


def copy_hardest_samples(new_dir, src_dir, hardest_sample_images):
    for src_paths in hardest_sample_images:
        source_path = os.path.join(src_dir, src_paths.split("\\")[2])

        dest_path = src_paths.split("\\")[2]
        dest_path = os.path.join(new_dir, dest_path)

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        src_file = os.path.join(source_path, src_paths.split("\\")[-1])
        dst_file = os.path.join(dest_path, src_paths.split("\\")[-1])
        shutil.copy(src_file, dst_file)


def get_hardest_samples(samples_dir):
    hardest_anchor_samples_c = hardest_samples(samples_dir + "/2023-03-17_11-24-16/Contour")
    hardest_anchor_samples_r = hardest_samples(samples_dir + "/2023-03-17_11-31-20/RGB")
    hardest_anchor_samples_t = hardest_samples(samples_dir + "/2023-03-17_11-47-31/Texture")

    return set(hardest_anchor_samples_c) | set(hardest_anchor_samples_r) | set(hardest_anchor_samples_t)


def get_unique_values(hardest_sample):
    samples = []
    for name in hardest_sample:
        file_name = name.split("\\")[-1]
        file_name = file_name.replace("contour_", "").replace("texture_", "")
        samples.append(file_name)
    return set(samples)


def files_to_move(hardest_sample_images, src_dir):
    list_of_files_to_move = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            copy_of_file = file
            copy_of_file = copy_of_file.replace("contour_", "").replace("texture_", "")
            if copy_of_file in hardest_sample_images:
                list_of_files_to_move.append(os.path.join(root, file))

    return list_of_files_to_move

def main():
    hardest_anchor_samples = get_hardest_samples(CONST.dir_hardest_anc_samples)
    hardest_positive_samples = get_hardest_samples(CONST.dir_hardest_pos_samples)
    hardest_negative_samples = get_hardest_samples(CONST.dir_hardest_neg_samples)

    anchor = get_unique_values(hardest_anchor_samples)
    pos = get_unique_values(hardest_positive_samples)
    neg = get_unique_values(hardest_negative_samples)

    hardest_sample_images = anchor | pos | neg

    files_to_move_contour = files_to_move(hardest_sample_images, CONST.dir_contour)
    copy_hardest_samples(CONST.dir_contour_hardest, CONST.dir_contour, files_to_move_contour)

    files_to_move_rgb = files_to_move(hardest_sample_images, CONST.dir_bounding_box)
    copy_hardest_samples(CONST.dir_rgb_hardest, CONST.dir_bounding_box, files_to_move_rgb)

    files_to_move_texture = files_to_move(hardest_sample_images, CONST.dir_texture)
    copy_hardest_samples(CONST.dir_texture_hardest, CONST.dir_texture, files_to_move_texture)


if __name__ == "__main__":
    main()
