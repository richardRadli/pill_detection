from config.data_paths import NLP_DATA_PATH

from typing import Dict


def nlp_configs() -> Dict:
    nlp_config = {
        "pill_names":
            NLP_DATA_PATH.get_data_path("pill_names"),
        "full_sentence_csv":
            NLP_DATA_PATH.get_data_path("full_sentence_csv"),
        "vector_distances":
            NLP_DATA_PATH.get_data_path("vector_distances"),
        "nlp_vector":
            NLP_DATA_PATH.get_data_path("nlp_vector"),
        "word_vector_vis":
            NLP_DATA_PATH.get_data_path("word_vector_vis"),
        "elbow":
            NLP_DATA_PATH.get_data_path("elbow"),
        "silhouette":
            NLP_DATA_PATH.get_data_path("silhouette"),
        "patient_information_leaflet_doc":
            NLP_DATA_PATH.get_data_path("patient_information_leaflet_doc"),
        "patient_information_leaflet_docx":
            NLP_DATA_PATH.get_data_path("patient_information_leaflet_docx"),
        "extracted_features_files":
            NLP_DATA_PATH.get_data_path("extracted_features_files")
    }

    return nlp_config
