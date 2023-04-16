import os
import shutil

from glob import glob

from const import CONST


def get_hardest_samples(samples_dir, sub_dir):
    directory = os.path.join(samples_dir, sub_dir)
    return max(glob(os.path.join(directory, "*.txt")), key=os.path.getctime)


def process_txt(txt_file):
    paths = []

    with open(txt_file, 'r') as f:
        data = eval(f.read())

    for key in data:
        paths.append(key)

    return set(paths)


def get_unique_values(hardest_sample):
    samples = []
    for name in hardest_sample:
        file_name = name.split("\\")[-1]
        file_name = file_name.replace("contour_", "").replace("texture_", "")
        samples.append(file_name)
    return samples


def files_to_move(hardest_sample_images, src_dir):
    list_of_files_to_move = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            copy_of_file = file
            copy_of_file = copy_of_file.replace("contour_", "").replace("texture_", "")
            if copy_of_file in hardest_sample_images:
                list_of_files_to_move.append(os.path.join(root, file))

    return list_of_files_to_move


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


def main():
    latest_contour_txt = get_hardest_samples(CONST.dir_hardest_neg_samples, "2023-04-16_15-37-26/Contour")
    latest_rgb_txt = get_hardest_samples(CONST.dir_hardest_neg_samples, "2023-04-16_15-17-19/RGB")
    latex_texture_txt = get_hardest_samples(CONST.dir_hardest_neg_samples, "2023-04-16_15-49-20/Texture")

    hardest_samples_contour = process_txt(latest_contour_txt)
    hardest_samples_rgb = process_txt(latest_rgb_txt)
    hardest_samples_texture = process_txt(latex_texture_txt)

    hardest_samples_union = hardest_samples_contour | hardest_samples_rgb | hardest_samples_texture

    result = {x.split('\\')[-1] for x in hardest_samples_union}
    files_to_move_contour = files_to_move(result, CONST.dir_contour)
    copy_hardest_samples(CONST.dir_contour_hardest, CONST.dir_contour, files_to_move_contour)

    files_to_move_rgb = files_to_move(result, CONST.dir_bounding_box)
    copy_hardest_samples(CONST.dir_rgb_hardest, CONST.dir_bounding_box, files_to_move_rgb)

    files_to_move_texture = files_to_move(result, CONST.dir_texture)
    copy_hardest_samples(CONST.dir_texture_hardest, CONST.dir_texture, files_to_move_texture)


if __name__ == "__main__":
    main()
