import os
import shutil

from glob import glob
from tqdm import tqdm

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
    for src_paths in tqdm(hardest_sample_images):
        source_path = os.path.join(src_dir, src_paths.split("\\")[2])

        dest_path = src_paths.split("\\")[2]
        dest_path = os.path.join(new_dir, dest_path)

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        src_file = os.path.join(source_path, src_paths.split("\\")[-1])
        dst_file = os.path.join(dest_path, src_paths.split("\\")[-1])
        shutil.copy(src_file, dst_file)


def main():
    latest_neg_contour_txt = get_hardest_samples(CONST.dir_hardest_neg_samples, "2023-04-17_15-32-47/Contour")
    latest_neg_rgb_txt = get_hardest_samples(CONST.dir_hardest_neg_samples,     "2023-04-17_15-36-07/RGB")
    latex_neg_texture_txt = get_hardest_samples(CONST.dir_hardest_neg_samples,  "2023-04-17_15-29-05/Texture")

    latest_pos_contour_txt = get_hardest_samples(CONST.dir_hardest_pos_samples, "2023-04-17_15-32-47/Contour")
    latest_pos_rgb_txt = get_hardest_samples(CONST.dir_hardest_pos_samples,     "2023-04-17_15-36-07/RGB")
    latex_pos_texture_txt = get_hardest_samples(CONST.dir_hardest_pos_samples,  "2023-04-17_15-29-05/Texture")

    hardest_neg_samples_contour = process_txt(latest_neg_contour_txt)
    hardest_neg_samples_rgb = process_txt(latest_neg_rgb_txt)
    hardest_neg_samples_texture = process_txt(latex_neg_texture_txt)

    hardest_pos_samples_contour = process_txt(latest_pos_contour_txt)
    hardest_pos_samples_rgb = process_txt(latest_pos_rgb_txt)
    hardest_pos_samples_texture = process_txt(latex_pos_texture_txt)

    hardest_neg_samples_union = hardest_neg_samples_contour | hardest_neg_samples_rgb | hardest_neg_samples_texture
    hardest_pos_samples_union = hardest_pos_samples_contour | hardest_pos_samples_rgb | hardest_pos_samples_texture
    hardest_samples_union = hardest_pos_samples_union | hardest_neg_samples_union

    result = {x.split('\\')[-1] for x in hardest_samples_union}

    files_to_move_contour = files_to_move(result, CONST.dir_contour)
    copy_hardest_samples(CONST.dir_contour_hardest, CONST.dir_contour, files_to_move_contour)

    files_to_move_rgb = files_to_move(result, CONST.dir_rgb)
    copy_hardest_samples(CONST.dir_rgb_hardest, CONST.dir_rgb, files_to_move_rgb)

    files_to_move_texture = files_to_move(result, CONST.dir_texture)
    copy_hardest_samples(CONST.dir_texture_hardest, CONST.dir_texture, files_to_move_texture)


if __name__ == "__main__":
    main()
