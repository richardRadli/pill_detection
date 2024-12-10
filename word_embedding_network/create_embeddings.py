import logging
import os
import torch

from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List

from config.dataset_paths_selector import dataset_images_path_selector
from config.networks_paths_selector import word_embedding_paths
from config.json_config import json_config_selector
from dataloader_word_embedding_network import DataLoaderWordEmbeddingNetwork
from fully_connected_network import FullyConnectedNetwork
from utils.utils import (create_euc_matrix_file, find_latest_file_in_latest_directory_word_emb,
                         find_latest_file_in_directory, plot_euclidean_distances, plot_embeddings, create_timestamp,
                         use_gpu_if_available, load_config_json)


class CreateEmbedding:
    def __init__(self, interactive, plot_euc):
        self.interactive = interactive
        self.plot_euc = plot_euc

        # Init config
        timestamp = create_timestamp()
        self.cfg = load_config_json(
            json_schema_filename=json_config_selector("word_embedding").get("schema"),
            json_filename=json_config_selector("word_embedding").get("config")
        )

        device = use_gpu_if_available()
        word_emb_model_config = word_embedding_paths(self.cfg.dataset_type)

        # Aux variables
        self.output_dict = {}
        self.neuron_split = self.cfg.get("neuron_split")

        # Load model
        self.model = self.load_model(word_emb_model_config, device)

        # Load Dataset
        self.dataloader, self.class_ids = self.load_dataset()

        # Filenames
        self.euc_mtx_filename = (
            self.create_filename(word_emb_model_config.get("emb_euc_mtx"),
                                 timestamp,
                                 "emb_euc_mtx.png")
        )
        self.emb_tsne_filename = (
            self.create_filename(word_emb_model_config.get("emb_tsne"),
                                 timestamp,
                                 "emb_tsne.png")
        )
        self.emb_mtx_xlsx_filename = (
            self.create_filename(
                dataset_images_path_selector(self.cfg.dataset_type).get("dynamic_margin").get("euc_mtx_xlsx"),
                timestamp,
                "emb_euc_mtx.xlsx")
        )

    def load_model(self, word_emb_model_config: Dict[str, Any], device):
        """
        Load the pre-trained word embedding model.

        Args:
            word_emb_model_config (Dict[str, Any]): Configuration for the word embedding model.
            device: Device to load the model onto (e.g., 'cpu', 'cuda').

        Returns:
            FullyConnectedNetwork: Loaded word embedding model.
        """

        model = (
            FullyConnectedNetwork(
                neurons=self.cfg.get("neurons")).to(device)
        )

        latest_pt_file = (
            find_latest_file_in_latest_directory_word_emb(word_emb_model_config.get("weights"))
        )

        model.load_state_dict(torch.load(latest_pt_file))
        model.eval()

        return model

    def load_dataset(self) -> Tuple[DataLoader, List[str]]:
        """
        Load the dataset for word embedding network training.

        Returns:
            Tuple[DataLoader, List[str]]: A tuple containing the DataLoader for the dataset and
                the list of class IDs.
        """

        dataset_paths = (
            dataset_images_path_selector(self.cfg.dataset_type)
        )

        json_file_path = (
            find_latest_file_in_directory(
                path=dataset_paths.get("dynamic_margin").get("concatenated_vectors"),
                extension="json"
            )
        )

        dataset = (
            DataLoaderWordEmbeddingNetwork(
                json_file_path,
                self.neuron_split
            )
        )

        dataloader = (
            DataLoader(
                dataset,
                batch_size=len(dataset),
                shuffle=False
            )
        )

        class_ids = dataset.classes

        return dataloader, class_ids

    @staticmethod
    def create_filename(root_dir: str, timestamp: str, filename: str) -> str:
        """
        Create a filename by concatenating the timestamp and filename.

        Args:
            root_dir (str): Root directory where the file will be located.
            timestamp (str): Timestamp to be included in the filename.
            filename (str): Name of the file.

        Returns:
            str: Generated filename including root directory, timestamp, and filename.
        """

        return os.path.join(str(root_dir), f"{timestamp}_{filename}")

    def embedder(self):
        """
       Generate embeddings for dataset samples using the loaded model.

       Returns:
           None
       """

        for original, _ in self.dataloader:
            original = original.to("cuda")
            output = self.model(original)

            output = output.detach().cpu().numpy()
            for output_embedding, class_label in zip(output, self.class_ids):
                class_label = str(class_label)
                if class_label not in self.output_dict:
                    self.output_dict[class_label] = []
                self.output_dict[class_label].append(output_embedding)

    def main(self) -> None:
        """

        Return:
             None
        """

        self.embedder()

        labels = []
        embedding_vectors = []
        for label, embedding_list in self.output_dict.items():
            for embedding in embedding_list:
                labels.append(label)
                embedding_vectors.append(embedding)

        # tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=500, random_state=42)
        # new_values = tsne_model.fit_transform(np.array(embedding_vectors))
        pca_model = PCA(n_components=2)

        new_values = (
            pca_model.fit_transform(
                embedding_vectors
            )
        )

        create_euc_matrix_file(
            matrix=new_values,
            list_of_labels=labels,
            file_name=self.emb_mtx_xlsx_filename
        )

        if self.plot_euc:
            plot_euclidean_distances(
                vectors=self.output_dict,
                dataset_name=self.cfg.dataset_type,
                filename=self.euc_mtx_filename,
                normalize=False,
                operation="embeddings",
                plot_size=80
            )

        plot_embeddings(
            output_dict=self.output_dict,
            output_filename=self.emb_tsne_filename,
            interactive=self.interactive
        )


if __name__ == "__main__":
    try:
        emb = CreateEmbedding(interactive=True, plot_euc=False)
        emb.main()
    except KeyboardInterrupt as kie:
        logging.error(f"{kie}")
