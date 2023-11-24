import os
import shutil

from tqdm import tqdm


def folds():
    return {
        "fold1": ['concor_5_mg', 'covercard_plus_10_mg_2_5_mg_5_mg', 'enterol_250_mg', 'escitil_10_mg',
                  'frontin_0_25_mg', 'lactamed', 'mezym_forte_10_000_egyseg', 'narva_sr_1_5_mg_retard', 'no_spa_40_mg',
                  'nurofen_forte_400_mg', 'provera_5_mg', 'semicillin_500_mg', 'teva_ambrobene_30_mg',
                  'teva_enterobene_2_mg', 'tricovel_tricoage45', 'xeter_20_mg'],
        "fold2": ['algoflex_forte_dolo_400_mg', 'algopyrin_500_mg', 'atoris_20_mg', 'cataflam_50_mg', 'coldrex',
                  'frontin_0_5_mg', 'magne_b6', 'meridian', 'nebivolol_sandoz_5_mg', 'neo_citran',
                  'neo_ferro_folgamma_114_mg_0_8_mg', 'normodipine_5_mg', 'ocutein', 'sinupret_forte',
                  'verospiron_25_mg', 'voltaren_dolo_rapid_25_mg'],
        "fold3": ['algoflex_rapid_400_mg', 'betaloc_50_mg', 'cataflam_dolo_25_mg', 'cetirizin_10_mg',
                  'doxazosin_sandoz_uro_4_mg', 'indastad_1_5_mg', 'ketodex_25_mg', 'koleszterin_kontroll',
                  'naprosyn_250_mg', 'naturland_d_vitamin_forte', 'noclaud_50_mg', 'pantoprazol_sandoz_40_mg',
                  'revicet_akut_10_mg', 'salazopyrin_en_500_mg', 'strepfen_8_75_mg', 'valeriana_teva'],
        "fold4": ['acc_long_600_mg', 'apranax_550_mg', 'aspirin_ultra_500_mg', 'c_vitamin_teva_500_mg',
                  'co_perineva_4_mg_1_25_mg', 'cold_fx', 'lordestin_5_mg', 'milgamma_n', 'milurit_300_mg',
                  'olicard_60_mg', 'rhinathiol_tusso_100_mg', 'sedatif_pc', 'sicor_10_mg', 'syncumar_mite_1_mg',
                  'tritace_hct_5_mg_25_mg', 'urzinol'],
        "fold5": ['ambroxol_egis_30_mg', 'atorvastatin_teva_20_mg', 'calci_kid', 'cataflam_v_50_mg', 'concor_10_mg',
                  'dulsevia_60_mg', 'jutavit_cink', 'kalcium_magnezium_cink', 'kalium_r',
                  'l_thyroxin_henning_50_mikrogramm', 'lactiv_plus', 'laresin_10_mg', 'letrox_50_mikrogramm',
                  'mebucain_mint_2_mg_1_mg', 'merckformin_xr_1000_mg', 'sirdalud_4_mg']
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
                "C:/Users/ricsi/Documents/project/storage/IVM/images/stream_images/ogyei/%s/%s" % (
                category_dir, sub_dirs_train
                )
            )
            print(src_path, dst_path)
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)


def erase_files():
    sub_dirs_trains = ["train", "valid"]
    sub_dirs_tests = ["ref", "query"]

    for _, (sub_dirs_train, sub_dirs_test) in tqdm(enumerate(zip(sub_dirs_trains, sub_dirs_tests)),
                                                   total=len(sub_dirs_trains)):
        src_path = "C:/Users/ricsi/Documents/project/storage/IVM/images/test/ogyei/%s/" % sub_dirs_test
        shutil.rmtree(src_path)


if __name__ == "__main__":
    move_images_to_folds("fold5", "train", "ref")
    move_images_to_folds("fold5", "valid", "query")
    clean_up_empty_dirs()
    move_hardest_samples()
    # rollback_folds()
    # erase_files()
