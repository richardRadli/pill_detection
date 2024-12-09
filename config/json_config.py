from config.data_paths import JSON_FILES_PATHS


def json_config_selector(operation):
    json_cfg = {
        "augmentation": {
            "config": JSON_FILES_PATHS.get_data_path("config_augmentation"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_augmentation")
        },
        "fusion_net": {
            "config": JSON_FILES_PATHS.get_data_path("config_fusion_net"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_fusion_net")
        },
        "stream_images": {
            "config": JSON_FILES_PATHS.get_data_path("config_stream_images"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_stream_images")
        },
        "stream_net": {
            "config": JSON_FILES_PATHS.get_data_path("config_stream_net"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_stream_net")
        },
        "word_embedding": {
            "config": JSON_FILES_PATHS.get_data_path("config_word_embedding"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_word_embedding")
        }
    }

    return json_cfg[operation]
