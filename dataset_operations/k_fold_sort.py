import os
import shutil

from tqdm import tqdm


def folds():
    return {
        "fold1": ['algoflex_forte_dolo', 'algopyrin', 'co_xeter', 'dorithricin_mentol', 'dulodet', 'dulsevia',
                  'koleszterin_kontroll', 'l_thyroxin', 'mebucaim_mint', 'merckformin_xr', 'mezym_forte',
                  'panangin', 'sicor', 'sirdalud4', 'superbrands_cink', 'teva_ambrobene'],
        "fold2": ['aspirin_ultra', 'atorvastatin_teva', 'betaloc', 'frontin', 'frontin_05', 'lordestin', 'milgamma_n',
                  'milurit', 'neocitran', 'nurofen_forte', 'olicard', 'semicillin', 'syncumar_mite', 'tricovel',
                  'xeter_20mg', 'zadex'],
        "fold3": ['advil_ultra_forte', 'algoflex_rapid', 'bila_git', 'cataflam', 'concor_5', 'condrosulf',
                  'dvitamin_forte', 'favipiravir', 'kalcium_magnezium_cink', 'ketodex', 'lactamed', 'magneb6',
                  'milgamma', 'novo_c_plus', 'pantoprazol_sandoz', 'urzinol'],
        "fold4": ['ambroxol_egis_30mg', 'atoris', 'coldrex', 'coverex', 'ibumax', 'kalium_r', 'lactiv_plus', 'no_spa',
                  'normodipine', 'ocutein', 'rhinathiol', 'rubophen_500mg', 'salazopyrin', 'superbrands_cvitamin',
                  'tritace', 'vitac'],
        "fold5": ['apranax', 'c_vitamin_teva', 'cataflam_v_50mg', 'concor_10', 'controloc', 'escitil', 'furon',
                  'meridian', 'neo_ferro_folgamma', 'tritace_htc', 'urotrin', 'valeriana_teva', 'vitamin_d',
                  'voltaren_dolo_rapid']
    }


def move_images_to_folds(fold_id: str = "fold1", op: str = "train", op2: str = "ref"):
    folders_to_copy = folds().get(fold_id)

    # Root source directory
    source_root = r'C:/Users/ricsi/Documents/project/storage/IVM/images/stream_images/ogyei/'

    # Destination root directory
    destination_root = r'C:/Users/ricsi/Documents/project/storage/IVM/images/test/ogyei/%s' % op2

    # Source and destination subdirectories
    source_subdirs = ['contour', 'lbp', 'rgb', 'texture']
    destination_subdirs = ['contour', 'lbp', 'rgb', 'texture']

    # Iterate through source and destination subdirectories
    for source_dir, dest_dir in zip(source_subdirs, destination_subdirs):
        source_dir = os.path.join(source_root, source_dir)
        dest_dir = os.path.join(destination_root, dest_dir)

        # Iterate through folders to copy
        for folder in folders_to_copy:
            source_path = os.path.join(source_dir, op, folder)
            dest_path = os.path.join(dest_dir, folder.lower())

            # Check if the source folder exists
            if os.path.exists(source_path):
                # Create destination directories if they don't exist
                os.makedirs(dest_path, exist_ok=True)

                # Copy the folder contents
                for item in os.listdir(source_path):
                    source_item = os.path.join(source_path, item)
                    dest_item = os.path.join(dest_path, item)
                    if os.path.isdir(source_item):
                        shutil.move(source_item, dest_item)
                    else:
                        shutil.move(source_item, dest_item)

                print(f"Copied {folder} from {source_path} to {dest_path}")
            else:
                print(f"Folder {folder} not found in {source_path}")


def delete_empty_subdirectories(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        for dirname in dirnames:
            dir_to_check = os.path.join(dirpath, dirname)
            if not os.listdir(dir_to_check):
                print(f"Deleting empty directory: {dir_to_check}")
                os.rmdir(dir_to_check)


def clean_up_empty_dirs():
    main_dirs = ['contour', 'lbp', 'rgb', 'texture']
    sub_dirs = ["train", "valid"]
    for main_dir in main_dirs:
        for sub_dir in sub_dirs:
            root_path = ("C:/Users/ricsi/Documents/project/storage/IVM/images/stream_images/ogyei/%s/%s" %
                         (main_dir, sub_dir))
            if os.path.exists(root_path):
                delete_empty_subdirectories(root_path)
                print("Empty subdirectories deleted.")
            else:
                print("Root path does not exist.")


def move_hardest_samples():
    main_dirs = ['contour', 'lbp', 'rgb', 'texture']
    main_dirs_2 = ['contour_hardest', 'lbp_hardest', 'rgb_hardest', 'texture_hardest']
    sub_dirs_train = ["train", "valid"]

    for _, (main_dir, main_dir_2) in tqdm(enumerate(zip(main_dirs, main_dirs_2)), total=len(main_dirs)):
        dest_path = (
                "C:/Users/ricsi/Documents/project/storage/IVM/images/hardest_samples/efficient_net/%s" % main_dir_2)
        for sub_dir_tr in tqdm(sub_dirs_train, total=len(sub_dirs_train)):
            source_path = \
                ("C:/Users/ricsi/Documents/project/storage/IVM/images/stream_images/ogyei/%s/%s" % (main_dir, sub_dir_tr))
            shutil.copytree(source_path, dest_path, dirs_exist_ok=True)


def rollback_folds():
    category_dirs = ['contour', 'lbp', 'rgb', 'texture']
    sub_dirs_trains = ["train", "valid"]
    sub_dirs_tests = ["ref", "query"]

    for _, (sub_dirs_train, sub_dirs_test) in tqdm(enumerate(zip(sub_dirs_trains, sub_dirs_tests)),
                                                   total=len(sub_dirs_trains)):
        for category_dir in category_dirs:
            src_path = (
                "C:/Users/ricsi/Documents/project/storage/IVM/images/test/ogyei/%s/%s" % (sub_dirs_test, category_dir))
            dst_path = (
                "C:/Users/ricsi/Documents/project/storage/IVM/images/stream_images/ogyei/%s/%s" % (category_dir, sub_dirs_train)
            )

            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)


if __name__ == "__main__":
    move_images_to_folds("fold5", "train", "ref")
    move_images_to_folds("fold5", "valid", "query")
    clean_up_empty_dirs()
    move_hardest_samples()

    # rollback_folds()
