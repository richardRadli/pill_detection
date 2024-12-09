"""
File: predict_stream_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: This code implements the inference for the stream network phase.
"""

import colorama
import logging
import os
import pandas as pd
import torch

from torchvision import transforms
from tqdm import tqdm
from typing import Tuple
from PIL import Image

from config.json_config import json_config_selector
from config.networks_paths_selector import substream_paths
from stream_network_models.stream_network_selector import StreamNetworkFactory
from utils.utils import (use_gpu_if_available, create_timestamp, find_latest_file_in_latest_directory,
                         plot_ref_query_images, setup_logger, load_config_json)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++ P R E D I C T   S T R E A M   N E T W O R K ++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PredictStreamNetwork:
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
        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("stream_net").get("schema"),
                json_filename=json_config_selector("stream_net").get("config")
            )
        )

        self.dataset_type = self.cfg.get("dataset_type")
        self.type_of_net = self.cfg.get("type_of_net")
        self.stream_network_type = self.cfg.get("type_of_stream")
        self.loss_type = self.cfg.get("type_of_loss_func")
        self.dmtl_type = self.cfg.get("type_of_dmtl")

        # Load networks
        self.network_con, self.network_lbp, self.network_rgb, self.network_tex = self.load_networks()

        # Preprocess images
        image_size = self.cfg.get("networks").get(self.cfg.get("type_of_net")).get("image_size")
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
                network_cfg=substream_paths().get(self.stream_network_type),
                subdir="plotting_folder",
                loss=self.loss_type,
                dmtl_type=self.dmtl_type
            )
        )

        self.prediction_dir = (
            self.create_save_dirs(
                network_cfg=substream_paths().get(self.stream_network_type),
                subdir="prediction_folder",
                loss=self.loss_type,
                dmtl_type=self.dmtl_type
            )
        )

        self.ref_save_dir = (
            self.create_save_dirs(
                network_cfg=substream_paths().get(self.stream_network_type),
                subdir="ref_vectors_folder",
                loss=self.loss_type,
                dmtl_type=self.dmtl_type
            )
        )

    def create_save_dirs(self, network_cfg, subdir, loss, dmtl_type) -> str:
        """
        Creates and returns a directory path based on the provided network configuration.

        Args:
            network_cfg: A dictionary containing network configuration information.
            subdir: The subdirectory to create the directory path for.
            loss: Loss type
            dmtl_type: Only set it when dmtl is used
        Returns:
            directory_to_create (str): The path of the created directory.
        """

        if loss == "hmtl":
            directory_path = network_cfg.get(subdir).get(loss)
        else:
            directory_path = network_cfg.get(subdir).get(loss).get(dmtl_type)

        directory_to_create = (
            os.path.join(directory_path, f"{self.timestamp}")
        )
        os.makedirs(directory_to_create, exist_ok=True)
        return directory_to_create

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- L O A D   N E T W O R K S -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def load_networks(self) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
        """
        Load pretrained networks using the latest .pt files.

        Returns:
            Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
                Contour, LBP, RGB, and Texture networks.
        """

        contour_substream_network_cfg = self.cfg.get("streams").get("Contour")
        lbp_substream_network_cfg = self.cfg.get("streams").get("LBP")
        rgb_substream_network_cfg = self.cfg.get("streams").get("RGB")
        texture_substream_network_cfg = self.cfg.get("streams").get("Texture")

        latest_con_pt_file = find_latest_file_in_latest_directory(
            path=contour_substream_network_cfg.get("model_weights_dir").get(self.loss_type) if self.loss_type == "hmtl"
            else contour_substream_network_cfg.get("model_weights_dir").get(self.loss_type).get(self.dmtl_type),
            type_of_loss=self.cfg.type_of_loss_func
        )
        latest_lbp_pt_file = find_latest_file_in_latest_directory(
            path=lbp_substream_network_cfg.get("model_weights_dir").get(self.loss_type) if self.loss_type == "hmtl" else
            lbp_substream_network_cfg.get("model_weights_dir").get(self.loss_type).get(self.dmtl_type),
            type_of_loss=self.cfg.type_of_loss_func
        )
        latest_rgb_pt_file = find_latest_file_in_latest_directory(
            path=rgb_substream_network_cfg.get("model_weights_dir").get(self.loss_type) if self.loss_type == "hmtl" else
            rgb_substream_network_cfg.get("model_weights_dir").get(self.loss_type).get(self.dmtl_type),
            type_of_loss=self.cfg.type_of_loss_func
        )
        latest_tex_pt_file = find_latest_file_in_latest_directory(
            path=texture_substream_network_cfg.get("model_weights_dir").get(
                self.loss_type) if self.loss_type == "hmtl" else
            texture_substream_network_cfg.get("model_weights_dir").get(self.loss_type).get(self.dmtl_type),
            type_of_loss=self.cfg.type_of_loss_func
        )

        network_con = StreamNetworkFactory.create_network(self.type_of_net, contour_substream_network_cfg)
        network_lbp = StreamNetworkFactory.create_network(self.type_of_net, lbp_substream_network_cfg)
        network_rgb = StreamNetworkFactory.create_network(self.type_of_net, rgb_substream_network_cfg)
        network_tex = StreamNetworkFactory.create_network(self.type_of_net, texture_substream_network_cfg)

        network_con.load_state_dict(torch.load(latest_con_pt_file))
        network_lbp.load_state_dict(torch.load(latest_lbp_pt_file))
        network_rgb.load_state_dict(torch.load(latest_rgb_pt_file))
        network_tex.load_state_dict(torch.load(latest_tex_pt_file))

        return network_con, network_lbp, network_rgb, network_tex

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- G E T   V E C T O R S ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_vectors(self, images_dirs: dict, operation: str):
        """
        Args:
            images_dirs:
                dictionary containing paths to the query/reference images for contour, LBP, RGB, and texture streams.
            operation:
                string indicating the type of operation ('query' or 'reference').

        Returns:
            tuple containing dictionaries - vectors, labels, and image_paths.
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

                # Forward pass through the networks (Query embeddings)
                with torch.no_grad():
                    contour_vector = self.network_con(contour_image)
                    lbp_vector = self.network_lbp(lbp_image)
                    rgb_vector = self.network_rgb(rgb_image)
                    texture_vector = self.network_tex(tex_image)

                # Concatenate the embeddings into one vector
                concatenated_vector = torch.cat([contour_vector, lbp_vector, rgb_vector, texture_vector], dim=1).cpu()

                # Append results to the dictionary
                vectors[image_name].append(concatenated_vector)
                images_path[image_name].append(os.path.join(images_dirs['rgb'], image_name, image_paths['rgb'][idx]))
                ground_truth_labels.append(image_name)

        logging.info(f"Processing of {operation} images is complete")
        return vectors, images_path, ground_truth_labels

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ S A V E   R E F E R E N C E   V E C T O R -----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_reference_vector(self, vectors, labels, images_path):
        """

            vectors:
            labels:
            images_path:

        Return:
            None
        """

        os.makedirs(self.ref_save_dir, exist_ok=True)
        torch.save({'vectors': vectors,
                    'labels': labels,
                    'images_path': images_path},
                   os.path.join(self.ref_save_dir, "ref_vectors.pt"))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------- M E A S U R E   C O S S I M   A N D   E U C D I S T ------------------------------
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

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- D I S P L A Y   R E S U L T S ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
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
                substream_paths().get("Contour").get(self.dataset_type).get(self.type_of_net).get("test").get("query"),
            "lbp":
                substream_paths().get("LBP").get(self.dataset_type).get(self.type_of_net).get("test").get("query"),
            "rgb":
                substream_paths().get("RGB").get(self.dataset_type).get(self.type_of_net).get("test").get("query"),
            "tex":
                substream_paths().get("Texture").get(self.dataset_type).get(self.type_of_net).get("test").get("query")
        }

        reference_dirs = {
            "con":
                substream_paths().get("Contour").get(self.dataset_type).get(self.type_of_net).get("test").get("ref"),
            "lbp":
                substream_paths().get("LBP").get(self.dataset_type).get(self.type_of_net).get("test").get("ref"),
            "rgb":
                substream_paths().get("RGB").get(self.dataset_type).get(self.type_of_net).get("test").get("ref"),
            "tex":
                substream_paths().get("Texture").get(self.dataset_type).get(self.type_of_net).get("test").get("ref")
        }

        query_vecs, query_image_paths, gt_labels = self.get_vectors(query_dirs, "query")
        reference_vecs, reference_image_paths, _ = self.get_vectors(reference_dirs, "reference")

        predicted_medicines = self.compare_query_and_reference_vectors(reference_vecs, query_vecs)
        self.display_results(query_vecs, predicted_medicines)

        # Plot query and reference medicines
        plot_dir = (
            os.path.join(
                self.plot_dir,
                f"{self.timestamp}"
            )
        )
        plot_ref_query_images(gt_labels, predicted_medicines, query_image_paths, reference_image_paths, plot_dir)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- __M A I N__ ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        pill_rec = PredictStreamNetwork()
        pill_rec.main()
    except KeyboardInterrupt as kie:
        logging.error(kie)
