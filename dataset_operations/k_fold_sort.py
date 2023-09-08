import os
import shutil


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


def move_images(fold_id: str = "fold1", op: str = "train", op2: str = "ref"):
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


def delete_images():
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


if __name__ == "__main__":
    # move_images("fold1", "train", "ref")
    # move_images("fold1", "valid", "query")
    delete_images()
