import torch
from torch.utils.data import DataLoader

from config.config import ConfigWordEmbedding
from config.config_selector import word_embedded_network_configs, dataset_images_path_selector
from dataloader_word_embedding_network import DataLoaderWordEmbeddingNetwork
from fully_connected_network import FullyConnectedNetwork
from utils.utils import (find_latest_file_in_latest_directory_word_emb, find_latest_file_in_directory,
                         plot_euclidean_distances)

cfg = ConfigWordEmbedding().parse()
word_emb_model_confing = word_embedded_network_configs(cfg.dataset_type)
model = FullyConnectedNetwork(input_dim=word_emb_model_confing.get("input_dim"),
                              hidden_dim=word_emb_model_confing.get("hidden_dim"),
                              output_dim=word_emb_model_confing.get("output_dim")).to("cuda")

latest_pt_file = find_latest_file_in_latest_directory_word_emb(word_emb_model_confing.get("model_weights_dir"))
model.load_state_dict(torch.load(latest_pt_file))
model.eval()

dataset_paths = dataset_images_path_selector(cfg.dataset_type)
json_file_path = (
    find_latest_file_in_directory(path=dataset_paths.get("dynamic_margin").get("concatenated_vectors"),
                                  extension="json")
)
dataset = DataLoaderWordEmbeddingNetwork(json_file_path)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
class_ids = dataset.classes

output_dict = {}
for original, _ in dataloader:
    original = original.to("cuda")
    output = model(original)

    output = output.detach().cpu().numpy()
    for output_embedding, class_label in zip(output, class_ids):
        class_label = str(class_label)
        if class_label not in output_dict:
            output_dict[class_label] = []
        output_dict[class_label].append(output_embedding)

filename = "C:/Users/ricsi/Desktop/asd.png"
plot_euclidean_distances(output_dict, cfg.dataset_type, filename, False, "embeddings", 40)
