import numpy as np
import os
import openpyxl

from sklearn.preprocessing import OneHotEncoder

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector


def encoding(words):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(np.array(words).reshape(-1, 1))
    return {word: encoded_features[idx].tolist() for idx, word in enumerate(words)}


def main():
    cfg = ConfigAugmentation().parse()

    imprint_words = ['PRINTED', 'DEBOSSED', 'EMBOSSED']
    score_words = [1, 2, 4]

    pill_desc_path = dataset_images_path_selector(cfg.dataset_name).get("other").get("pill_desc_xlsx")
    pill_desc_file = os.path.join(pill_desc_path, "pill_desc.xlsx")

    workbook = openpyxl.load_workbook(pill_desc_file)
    sheet = workbook['Sheet1']

    encoded_features_dict = encoding(imprint_words)
    encoded_scores_dict = encoding(score_words)

    imprint_dict = {}
    scores_dict = {}

    for row in sheet.iter_rows(min_row=3, values_only=True):
        pill_id = row[1]
        imprint_type = row[4]
        score_type = row[7]
        imprint_dict[pill_id] = encoded_features_dict.get(imprint_type)
        scores_dict[pill_id] = encoded_scores_dict.get(score_type)

    print(imprint_dict, "\n", scores_dict)


if __name__ == '__main__':
    main()
