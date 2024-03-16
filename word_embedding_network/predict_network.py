import os
import logging
import torch

from config.config import ConfigWordEmbedding
from config.config_selector import word_embedded_network_configs, dataset_images_path_selector
from fully_connected_network import FullyConnectedNetwork
from utils.utils import (use_gpu_if_available, find_latest_file_in_latest_directory_word_emb,
                         find_latest_file_in_directory)
from word_embedding_network.dataloader_word_embedding_network import DataLoaderWordEmbeddingNetwork


class PredictNetwork:
    def __init__(self):
        self.cfg = ConfigWordEmbedding().parse()
        self.word_emb_model_confing = word_embedded_network_configs(self.cfg.dataset_type)
        self.device = use_gpu_if_available()
        self.model = self.load_network()
        self.model.eval()
        self.model.to(self.device)

        dataset_paths = dataset_images_path_selector(self.cfg.dataset_type)
        json_file_path = (
            find_latest_file_in_directory(path=dataset_paths.get("dynamic_margin").get("concatenated_vectors"),
                                          extension="json")
        )

        dataset = DataLoaderWordEmbeddingNetwork(json_file_path)
        self.classes = dataset.get_classes()

    def load_network(self):
        latest_pt_file = (
            find_latest_file_in_latest_directory_word_emb(self.word_emb_model_confing.get("model_weights_dir"))
        )
        network = FullyConnectedNetwork(input_dim=self.word_emb_model_confing.get("input_dim"),
                                        hidden_dim=self.word_emb_model_confing.get("hidden_dim"),
                                        output_dim=self.word_emb_model_confing.get("output_dim"))
        network.load_state_dict(torch.load(latest_pt_file))

        return network

    def predict(self):
        dummy = [
            1.0,
            0.13615886565620589,
            0.1293370766709946,
            0.8586676138912911,
            0.19064482790489343,
            0.0,
            0.0,
            0.6779434096645017,
            0.33589888276543606,
            1.0,
            0.6869901740008841,
            0.5971340142015139,
            0.7216150733998453,
            0.6168254247840168,
            0.3833651089507062,
            0.5957378742651864,
            0.6083781927907547,
            0.47527844869148045,
            0.5249579193003356,
            0.6037133333409948,
            0.5684418129212402,
            0.7048100874135012,
            0.5323462191575487,
            0.5133169845579805,
            0.6157206212488778,
            0.696334921091088,
            0.46548523112792006,
            0.655836735979971,
            0.5653680581776577,
            0.5587831138076346,
            0.619870379099973,
            0.5623440111932635,
            0.5836972482208125,
            0.5637571441047121,
            0.6177774515431984,
            0.61586577836413,
            0.556763473358355,
            0.5318283194878406,
            0.5137890598471583,
            0.5766175355344445,
            0.5656287697208009,
            0.59641545491872,
            0.5995923146154944,
            0.5619539561227946,
            0.5571485278471312,
            0.580347914697447,
            0.5186417855018717,
            0.6168728947171971,
            0.5315634991898758,
            0.5917768135634417,
            0.5972713552557879,
            0.5463543029213688,
            0.5963023789046489,
            0.5703743382526424,
            0.5949285367564873,
            0.5985992578431804,
            0.5693600863974992,
            0.5984633134926837,
            0.5625073726934443,
            0.5417644231606186,
            0.5714715708554498,
            0.6308323014340527,
            0.5459758156147634,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0
        ]

        dummy = torch.tensor(dummy)
        dummy = torch.reshape(dummy, (-1, 70))
        dummy = dummy.to(self.device)
        pred, emb = self.model(dummy)
        class_id = (torch.argmax(pred, dim=1))
        print(self.classes[class_id])


if __name__ == '__main__':
    predict_network = PredictNetwork()
    predict_network.predict()
