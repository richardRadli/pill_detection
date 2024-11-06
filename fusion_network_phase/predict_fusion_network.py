"""
File: predict_fusion_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 18, 2023

Description: This program implements the prediction for fusion networks.
"""

import colorama
import logging
import os
import pandas as pd
import torch

from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from config.json_config import json_config_selector
from config.networks_paths_selector import substream_paths
from config.networks_paths_selector import fusion_network_paths
from fusion_network_models.fusion_network_selector import FusionNetworkFactory
from utils.utils import (use_gpu_if_available, create_timestamp, find_latest_file_in_latest_directory,
                         plot_ref_query_images, setup_logger, load_config_json)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++ P R E D I C T   F U S I O N   N E T W O R K ++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PredictFusionNetwork:
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- __I N I T__ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.timestamp = create_timestamp()
        setup_logger()
        colorama.init()

        self.top5_indices = []
        self.confidence_percentages = None
        self.accuracy_top5 = None
        self.accuracy_top1 = None
        self.num_correct_top5 = 0
        self.num_correct_top1 = 0

        # Set up class variables
        self.preprocess_rgb = None
        self.preprocess_con_tex_lbp = None
        self.query_image_tex = None
        self.query_image_rgb = None
        self.query_image_con = None

        # Load config
        self.cfg_fusion_net = load_config_json(
                json_schema_filename=json_config_selector("fusion_net").get("schema"),
                json_filename=json_config_selector("fusion_net").get("config")
            )
        self.cfg_stream_net = (
            load_config_json(
                json_schema_filename=json_config_selector("stream_net").get("schema"),
                json_filename=json_config_selector("stream_net").get("config")
            )
        )

        self.dataset_type = self.cfg_fusion_net.get("dataset_type")
        self.fusion_network_type = self.cfg_fusion_net.get("type_of_net")
        self.loss_type = self.cfg_fusion_net.get("type_of_loss_func")

        # Load networks
        self.network = self.load_networks()

        # Preprocess images
        image_size = self.cfg_stream_net.get("networks").get(self.cfg_stream_net.get("type_of_net")).get("image_size")
        self.preprocess_rgb = \
            transforms.Compose(
                [
                    transforms.Resize(
                        (
                            image_size,
                            image_size
                        )
                    ),
                    transforms.CenterCrop(
                        (
                            image_size,
                            image_size
                        )
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )

        self.preprocess_con_tex_lbp = \
            transforms.Compose(
                [
                    transforms.Resize(
                        (
                            image_size,
                            image_size
                        )
                    ),
                    transforms.CenterCrop(
                        (
                            image_size,
                            image_size
                        )
                    ),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                ]
            )

        # Select device
        self.device = use_gpu_if_available()

        self.plot_dir = (
            self.create_save_dirs(
                    network_cfg=fusion_network_paths(dataset_type=self.dataset_type, network_type=self.fusion_network_type),
                    subdir="plotting_folder",
                    loss=self.loss_type
            )
        )

        self.prediction_dir = (
            self.create_save_dirs(
                    network_cfg=fusion_network_paths(dataset_type=self.dataset_type, network_type=self.fusion_network_type),
                    subdir="prediction_folder",
                    loss=self.loss_type
            )
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   S A V E   D I R S -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_save_dirs(self, network_cfg, subdir, loss) -> str:
        """
        Creates and returns a directory path based on the provided network configuration.

        Args:
            network_cfg: A dictionary containing network configuration information.
            subdir: The subdirectory to create the directory path for.
            loss: Loss type
        Returns:
            directory_to_create (str): The path of the created directory.
        """

        directory_path = network_cfg.get(subdir).get(loss)
        directory_to_create = (
            os.path.join(directory_path, f"{self.timestamp}")
        )
        os.makedirs(directory_to_create, exist_ok=True)
        return directory_to_create

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- L O A D   N E T W O R K S -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def load_networks(self) -> torch.nn.Module:
        """
        Load pretrained networks using the latest .pt files.

        Returns:
            network_fusion (torch.nn.Module): Fusion network.
        """

        latest_con_pt_file = find_latest_file_in_latest_directory(
            fusion_network_paths(
                dataset_type=self.dataset_type,
                network_type=self.fusion_network_type
            ).get("weights_folder").get(self.loss_type)
        )
        network_fusion = FusionNetworkFactory.create_network(fusion_network_type=self.fusion_network_type)
        network_fusion.load_state_dict(torch.load(latest_con_pt_file))
        network_fusion.network_con.eval()
        network_fusion.network_lbp.eval()
        network_fusion.network_rgb.eval()
        network_fusion.network_tex.eval()
        network_fusion.eval()

        return network_fusion

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- G E T   V E C T O R S ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_vectors(self, images_dirs: dict, operation: str):
        """
        Get feature vectors for images.

        Args:


        Returns:
             tuple containing three lists - vectors, labels, and images_path
        """

        logging.info(f"Processing {operation} images")
        color = colorama.Fore.BLUE if operation == "query" else colorama.Fore.RED
        medicine_classes = os.listdir(images_dirs["rgb"])

        vectors = {}
        labels = {}
        images_path = {}
        ground_truth_labels = []

        for image_name in tqdm(medicine_classes, desc=color + f"\nProcessing {operation} images"):
            # Collecting image paths for each stream

            image_paths = {
                'con': os.listdir(os.path.join(images_dirs['con'], image_name)),
                'lbp': os.listdir(os.path.join(images_dirs['lbp'], image_name)),
                'rgb': os.listdir(os.path.join(images_dirs['rgb'], image_name)),
                'tex': os.listdir(os.path.join(images_dirs['tex'], image_name))
            }

            vectors[image_name] = []
            labels[image_name] = []
            images_path[image_name] = []

            for idx, (con, lbp, rgb, tex) in enumerate(zip(
                    image_paths['con'], image_paths['lbp'], image_paths['rgb'], image_paths['tex'])
            ):
                # Load images and preprocess them
                contour_image = (
                    self.preprocess_con_tex_lbp(
                        Image.open(os.path.join(images_dirs['con'], image_name, con))
                    )
                )
                lbp_image = (
                    self.preprocess_con_tex_lbp(
                        Image.open(os.path.join(images_dirs['lbp'], image_name, lbp))
                    )
                )
                rgb_image = (
                    self.preprocess_rgb(
                        Image.open(os.path.join(images_dirs['rgb'], image_name, rgb))
                    )
                )
                tex_image = (
                    self.preprocess_con_tex_lbp(
                        Image.open(os.path.join(images_dirs['tex'], image_name, tex))
                    )
                )

                # Move to device
                contour_image, lbp_image, rgb_image, tex_image = [
                    img.unsqueeze(0).to(self.device) for img in
                    [contour_image, lbp_image, rgb_image, tex_image]
                ]

                with torch.no_grad():
                    # Move input to GPU
                    vector = self.network(contour_image, lbp_image, rgb_image, tex_image).cpu()

                vectors[image_name].append(vector)
                images_path[image_name].append(os.path.join(images_dirs['rgb'], image_name, image_paths['rgb'][idx]))
                ground_truth_labels.append(image_name)

        return vectors, images_path, ground_truth_labels

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------- M E A S U R E   S I M I L A R I T Y   A N D   D I S T A N C E -------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compare_query_and_reference_vectors(self, reference_vectors: dict, query_vectors: dict):
        """
        This method measures the similarity and distance between the query_vectors and all reference_vectors using
        Euclidean distance. It returns the predicted medicine labels based on the closest reference vector, and
        calculates top-1 and top-5 accuracy.

        Args:
            reference_vectors: a dictionary of embedded vectors for the reference set.
            query_vectors: a dictionary of embedded vectors for the query set.

        Returns:
            predicted_medicine_euc_dist: List of predicted labels for each query vector.
            similarity_scores_euc_dist: List of similarity scores for each query vector.
            most_similar_indices_euc_dist: List of indices of the most similar reference vectors.
            accuracy_top1: The top-1 accuracy.
            accuracy_top5: The top-5 accuracy.
        """

        logging.info("Comparing query and reference vectors")

        similarity_scores_euc_dist = []
        predicted_medicine_euc_dist = []
        most_similar_indices_euc_dist = []

        # Flatten all reference vectors into one tensor and track labels
        all_reference_vectors = []
        all_reference_labels = []

        for label, vectors in reference_vectors.items():
            all_reference_vectors.extend([vec.squeeze(0) for vec in vectors])
            all_reference_labels.extend([label] * len(vectors))  # Track the label for each vector

        all_reference_vectors_tensor = torch.stack(
            [torch.as_tensor(vec).to(self.device) for vec in all_reference_vectors])

        total_queries = 0

        for image_name, query_vector_list in tqdm(query_vectors.items(), desc="Comparing process"):

            total_queries += len(query_vector_list)
            query_vectors_tensor = torch.stack(
                [torch.as_tensor(vec).squeeze(0).to(self.device) for vec in query_vector_list]
            )

            for idx_query, query_vector in enumerate(query_vectors_tensor):
                scores_euclidean_distance = torch.norm(query_vector - all_reference_vectors_tensor, dim=1)

                # Get the index of the most similar reference vector
                most_similar_index = scores_euclidean_distance.argmin().item()
                most_similar_indices_euc_dist.append(most_similar_index)

                # Get the predicted medicine label (top-1 prediction)
                predicted_medicine = all_reference_labels[most_similar_index]
                predicted_medicine_euc_dist.append(predicted_medicine)

                # Check if the top-1 prediction is correct
                if predicted_medicine == image_name:
                    self.num_correct_top1 += 1

                # Get the top-5 predicted medicines
                top5_indices = torch.argsort(scores_euclidean_distance)[:5]
                top5_predicted_medicines = [all_reference_labels[i] for i in top5_indices]

                # Check if the correct label is in the top-5 predictions
                if image_name in top5_predicted_medicines:
                    self.num_correct_top5 += 1

                # Track the similarity scores for analysis if needed
                similarity_scores_euc_dist.append(scores_euclidean_distance.cpu().tolist())

        # Calculate accuracies
        self.accuracy_top1 = self.num_correct_top1 / total_queries
        self.accuracy_top5 = self.num_correct_top5 / total_queries

        return predicted_medicine_euc_dist

    def display_results(self, query_vectors: dict, predicted_labels: list) -> None:
        """
        Display the results of the prediction.

        Args:
            query_vectors: dict, containing the query vectors (ground truth labels are the keys of the dictionary).
            predicted_labels: list, predicted labels for the queries.

        Returns:
            None
        """

        # Extract the ground truth labels from the query_vectors dictionary keys
        ground_truth_labels = [label for label, vectors in query_vectors.items() for _ in vectors]

        # Ensure the number of ground truth labels matches the predicted labels
        if len(ground_truth_labels) != len(predicted_labels):
            raise ValueError("The number of ground truth labels and predicted labels must match!")

        # Create dataframe with the results
        df = (
            pd.DataFrame(
                list(zip(ground_truth_labels, predicted_labels)),
                columns=['GT Medicine Name', 'Predicted Medicine Name (ED)']
            )
        )

        # Create statistics dataframe
        df_stat = [
            ["Correctly predicted (Top-1):", f'{self.num_correct_top1}'],
            ["Correctly predicted (Top-5):", f'{self.num_correct_top5}'],
            ["Miss predicted top 1:", f'{len(ground_truth_labels) - self.num_correct_top1}'],
            ["Miss predicted top 5:", f'{len(ground_truth_labels) - self.num_correct_top5}'],
            ['Accuracy (Top-1):', f'{self.accuracy_top1:.4%}'],
            ['Accuracy (Top-5):', f'{self.accuracy_top5:.4%}']
        ]
        df_stat = pd.DataFrame(df_stat, columns=['Metric', 'Value'])

        # Set Pandas options for better visibility in logs or console
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        # Log and print the result
        logging.info(df)
        logging.info(df_stat)

        # Combine dataframes and save to a file
        df_combined = pd.concat([df, df_stat], ignore_index=True)
        df_combined.to_csv(
            os.path.join(self.prediction_dir,
                         f"{self.timestamp}_stream_network_prediction.txt"),
            sep='\t', index=True
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------- M A I N ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def main(self) -> None:
        """
        Executes the pipeline for prediction.

        Returns:
             None
        """

        query_dirs = {
            "con":
                substream_paths().get("Contour").get(self.dataset_type).get("EfficientNetV2").get("test").get("query"),
            "lbp":
                substream_paths().get("LBP").get(self.dataset_type).get("EfficientNetV2").get("test").get("query"),
            "rgb":
                substream_paths().get("RGB").get(self.dataset_type).get("EfficientNetV2").get("test").get("query"),
            "tex":
                substream_paths().get("Texture").get(self.dataset_type).get("EfficientNetV2").get("test").get("query")
        }

        reference_dirs = {
            "con":
                substream_paths().get("Contour").get(self.dataset_type).get("EfficientNetV2").get("test").get("ref"),
            "lbp":
                substream_paths().get("LBP").get(self.dataset_type).get("EfficientNetV2").get("test").get("ref"),
            "rgb":
                substream_paths().get("RGB").get(self.dataset_type).get("EfficientNetV2").get("test").get("ref"),
            "tex":
                substream_paths().get("Texture").get(self.dataset_type).get("EfficientNetV2").get("test").get("ref")
        }

        query_vecs, query_image_paths, gt_labels = self.get_vectors(query_dirs, "query")
        reference_vecs, reference_image_paths, _ = self.get_vectors(reference_dirs, "reference")

        predicted_medicines = self.compare_query_and_reference_vectors(reference_vecs, query_vecs)

        self.display_results(query_vecs, predicted_medicines)

        plot_ref_query_images(gt_labels, predicted_medicines, query_image_paths, reference_image_paths, self.plot_dir)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- __M A I N__ -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        pfn = PredictFusionNetwork()
        pfn.main()
    except KeyboardInterrupt as kie:
        print(kie)
