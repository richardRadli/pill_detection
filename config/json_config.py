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
            "config": JSON_FILES_PATHS.get_data_path("config_streamnet"),
            "schema": JSON_FILES_PATHS.get_data_path("config_schema_streamnet")
        }
    }

    return json_cfg[operation]
