"""
File: predict_stream_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: This code implements the inference for the stream network phase.
"""

import colorama
import logging
import numpy as np
import os
import pandas as pd
import torch

from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from tqdm import tqdm
from typing import List, Tuple
from PIL import Image

from config.config import ConfigStreamNetwork
from config.config_selector import sub_stream_network_configs, stream_network_config
from stream_network_models.stream_network_selector import NetworkFactory
from utils.utils import create_timestamp, find_latest_file_in_latest_directory, \
    plot_ref_query_images, use_gpu_if_available, setup_logger


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++ P R E D I C T   S T R E A M   N E T W O R K ++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PredictStreamNetwork:
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- __I N I T__ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        # Setup logger
        setup_logger()

        # Load config
        self.cfg = ConfigStreamNetwork().parse()

        # Set up tqdm colours
        colorama.init()

        # Create time stamp
        self.timestamp = create_timestamp()

        self.preprocess_rgb = None
        self.preprocess_con_tex = None

        self.accuracy_top5 = None
        self.accuracy_top1 = None
        self.num_correct_top1 = 0
        self.num_correct_top5 = 0
        self.top5_indices = []
        self.confidence_percentages = None

        # Load configs
        self.main_network_config = stream_network_config(self.cfg)
        self.sub_network_config = sub_stream_network_configs(self.cfg)

        # Load networks
        self.network_rgb= self.load_networks()
        # self.network_con.eval()
        # self.network_lbp.eval()
        self.network_rgb.eval()
        # self.network_tex.eval()

        # Preprocess images
        self.preprocess_rgb = \
            transforms.Compose(
                [
                    transforms.Resize((
                        self.sub_network_config.get(self.cfg.type_of_stream).get("image_size"),
                        self.sub_network_config.get(self.cfg.type_of_stream).get("image_size"))
                    ),
                    transforms.CenterCrop((
                    self.sub_network_config.get(self.cfg.type_of_stream).get("image_size"),
                    self.sub_network_config.get(self.cfg.type_of_stream).get("image_size"))
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )

        self.preprocess_con_tex = \
            transforms.Compose(
                [
                    transforms.Resize((
                        self.sub_network_config.get(self.cfg.type_of_stream).get("image_size"),
                        self.sub_network_config.get(self.cfg.type_of_stream).get("image_size"))
                    ),
                    transforms.CenterCrop((
                    self.sub_network_config.get(self.cfg.type_of_stream).get("image_size"),
                    self.sub_network_config.get(self.cfg.type_of_stream).get("image_size"))
                    ),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                ]
            )

        # Select device
        self.device = use_gpu_if_available()

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- L O A D   N E T W O R K S -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def load_networks(self):
        """
        This function loads the pretrained networks, with the latest .pt files

        :return: The contour, rgb, and texture networks.
        """

        con_config = self.sub_network_config.get("Contour")
        lbp_config = self.sub_network_config.get("LBP")
        rgb_config = self.sub_network_config.get("RGB")
        tex_config = self.sub_network_config.get("Texture")

        # latest_con_pt_file = find_latest_file_in_latest_directory(
        #     path=con_config.get("model_weights_dir").get(self.cfg.type_of_net).get(self.cfg.dataset_type),
        #     type_of_loss=self.cfg.type_of_loss_func
        # )
        # latest_lbp_pt_file = find_latest_file_in_latest_directory(
        #     path=lbp_config.get("model_weights_dir").get(self.cfg.type_of_net).get(self.cfg.dataset_type),
        #     type_of_loss=self.cfg.type_of_loss_func
        # )
        latest_rgb_pt_file = find_latest_file_in_latest_directory(
            path=rgb_config.get("model_weights_dir").get(self.cfg.type_of_net).get(self.cfg.dataset_type),
            type_of_loss=self.cfg.type_of_loss_func
        )
        # latest_tex_pt_file = find_latest_file_in_latest_directory(
        #     path=tex_config.get("model_weights_dir").get(self.cfg.type_of_net).get(self.cfg.dataset_type),
        #     type_of_loss=self.cfg.type_of_loss_func
        # )

        # network_con = NetworkFactory.create_network(self.cfg.type_of_net, con_config)
        # network_lbp = NetworkFactory.create_network(self.cfg.type_of_net, lbp_config)
        network_rgb = NetworkFactory.create_network(self.cfg.type_of_net, rgb_config)
        # network_tex = NetworkFactory.create_network(self.cfg.type_of_net, tex_config)

        # network_con.load_state_dict(torch.load(latest_con_pt_file))
        # network_lbp.load_state_dict(torch.load(latest_lbp_pt_file))
        network_rgb.load_state_dict(torch.load(latest_rgb_pt_file))
        # network_tex.load_state_dict(torch.load(latest_tex_pt_file))

        # return network_con, network_lbp, network_rgb, network_tex
        return network_rgb

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- G E T   V E C T O R S ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_vectors(self, contour_dir: str, lbp_dir: str, rgb_dir: str, texture_dir: str, operation: str):
        """
        :param contour_dir: path to the directory containing contour images
        :param lbp_dir: path to the directory containing LBP images
        :param rgb_dir: path to the directory containing rgb images
        :param texture_dir: path to the directory containing texture images
        :param operation: name of the operation being performed
        :return: tuple containing three lists - vectors, labels, and images_path
        """

        logging.info("Processing %s images" % operation)

        color = colorama.Fore.BLUE if operation == "query" else colorama.Fore.RED
        medicine_classes = os.listdir(rgb_dir)
        vectors = []
        labels = []
        images_path = []

        # Move the model to the GPU
        # self.network_con = self.network_con.to(self.device)
        # self.network_lbp = self.network_lbp.to(self.device)
        self.network_rgb = self.network_rgb.to(self.device)
        # self.network_tex = self.network_tex.to(self.device)

        for image_name in tqdm(medicine_classes, desc=color + "\nProcess %s images" % operation):
            # Collecting images
            # image_paths_con = os.listdir(os.path.join(contour_dir, image_name))
            image_paths_rgb = os.listdir(os.path.join(rgb_dir, image_name))
            # image_paths_tex = os.listdir(os.path.join(texture_dir, image_name))
            # image_paths_lbp = os.listdir(os.path.join(lbp_dir, image_name))

            for idx, rgb in enumerate(image_paths_rgb):
                # Open images and convert them to tensors
                # con_image = Image.open(os.path.join(contour_dir, image_name, con))
                # con_image = self.preprocess_con_tex(con_image)
                #
                # lbp_image = Image.open(os.path.join(lbp_dir, image_name, lbp))
                # lbp_image = self.preprocess_con_tex(lbp_image)

                rgb_image = Image.open(os.path.join(rgb_dir, image_name, rgb))
                images_path.append(os.path.join(rgb_dir, image_name, rgb))
                rgb_image = self.preprocess_rgb(rgb_image)

                # tex_image = Image.open(os.path.join(texture_dir, image_name, tex))
                # tex_image = self.preprocess_con_tex(tex_image)

                # Make prediction
                with torch.no_grad():
                    # Move input to GPU
                    # con_image = con_image.unsqueeze(0).to(self.device)
                    # lbp_image = lbp_image.unsqueeze(0).to(self.device)
                    rgb_image = rgb_image.unsqueeze(0).to(self.device)
                    # tex_image = tex_image.unsqueeze(0).to(self.device)

                    # Perform computation on GPU and move result back to CPU
                    # vector1 = self.network_con(con_image).squeeze().cpu()
                    # vector2 = self.network_lbp(lbp_image).squeeze().cpu()
                    vector3 = self.network_rgb(rgb_image).squeeze().cpu()
                    # vector4 = self.network_tex(tex_image).squeeze().cpu()

                # concatenated = torch.cat((vector1, vector2, vector3, vector4), dim=0)
                vectors.append(vector3)
                labels.append(image_name)

            if operation == "reference":
                ref_save_dir = (
                    os.path.join(
                        self.main_network_config.get('ref_vectors_folder').get(self.cfg.dataset_type),
                        f"{self.timestamp}_{self.cfg.type_of_loss_func}"
                    )
                )
                os.makedirs(ref_save_dir, exist_ok=True)
                torch.save({'vectors': vectors, 'labels': labels, 'images_path': images_path},
                           os.path.join(ref_save_dir, "ref_vectors.pt"))

        logging.info("Processing of %s images is done" % operation)
        return vectors, labels, images_path

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------- M E A S U R E   C O S S I M   A N D   E U C D I S T ------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compare_query_and_reference_vectors(self, q_labels: list[str], r_labels: list[str], reference_vectors: list,
                                            query_vectors: list) -> Tuple[List[str], List[str], List[int]]:
        """
        This method measures the similarity and distance between two sets of labels (q_labels and r_labels) and their
        corresponding embedded vectors (query_vectors and reference_vectors) using Euclidean distance.
        It returns the original query labels, predicted medicine labels, indices of the most similar medicines in the
        reference set.

        :param q_labels: a list of ground truth medicine names
        :param r_labels: a list of reference medicine names
        :param reference_vectors: a numpy array of embedded vectors for the reference set
        :param query_vectors: a numpy array of embedded vectors for the query set
        :return: the original query labels, predicted medicine labels, indices of the most similar medicines in the
        reference set, and mAP
        """

        logging.info("Comparing query and reference vectors")

        similarity_scores_euc_dist = []
        predicted_medicine_euc_dist = []
        corresp_sim_euc_dist = []
        most_similar_indices_euc_dist = []

        # Move vectors to GPU
        reference_vectors_tensor = torch.stack([torch.as_tensor(vec).to(self.device) for vec in reference_vectors])
        query_vectors_tensor = torch.stack([torch.as_tensor(vec).to(self.device) for vec in query_vectors])

        for idx_query, query_vector in tqdm(enumerate(query_vectors_tensor), total=len(query_vectors_tensor),
                                            desc="Comparing process"):
            scores_euclidean_distance = torch.norm(query_vector - reference_vectors_tensor, dim=1)

            # Move scores to CPU for further processing
            similarity_scores_euc_dist.append(scores_euclidean_distance.cpu().tolist())

            # Calculate and store the most similar reference vector, predicted medicine label, and corresponding
            # minimum Euclidean distance for each query vector
            most_similar_indices_euc_dist = [scores.index(min(scores)) for scores in similarity_scores_euc_dist]
            predicted_medicine = r_labels[most_similar_indices_euc_dist[idx_query]]
            predicted_medicine_euc_dist.append(predicted_medicine)

            most_similar_indices_and_scores_e = [(i, min(scores)) for i, scores in
                                                 enumerate(similarity_scores_euc_dist)]
            corresp_sim_euc_dist.append(most_similar_indices_and_scores_e[idx_query][1])

            # Calculate top-1 accuracy
            if predicted_medicine == q_labels[idx_query]:
                self.num_correct_top1 += 1

            # Calculate top-5 accuracy
            top5_predicted_medicines = [r_labels[i] for i in torch.argsort(scores_euclidean_distance)[:5]]
            if q_labels[idx_query] in top5_predicted_medicines:
                self.num_correct_top5 += 1

        self.accuracy_top1 = self.num_correct_top1 / len(query_vectors)
        self.accuracy_top5 = self.num_correct_top5 / len(query_vectors)

        # Calculate confidence
        confidence_percentages = [1 - (score / max(scores)) for score, scores in
                                  zip(corresp_sim_euc_dist, similarity_scores_euc_dist)]
        self.confidence_percentages = [cp * 100 for cp in confidence_percentages]

        # Find index position of the ground truth medicine
        for idx_query, query_label in enumerate(q_labels):
            top5_predicted_medicines = [r_labels[i] for i in np.argsort(similarity_scores_euc_dist[idx_query])[:5]]
            if query_label in top5_predicted_medicines:
                index = top5_predicted_medicines.index(query_label)
            else:
                index = -1
            self.top5_indices.append(index)

        return q_labels, predicted_medicine_euc_dist, most_similar_indices_euc_dist

    def compare_query_ref_vectors_knn(self, q_labels: List[str], r_labels: List[str], reference_vectors: list,
                                      query_vectors: list, n_neighbors: int = 5) \
            -> Tuple[List[str], List[str], List[int]]:
        """
        This method measures the k-Nearest Neighbors between two sets of labels (q_labels and r_labels) and their
        corresponding embedded vectors (query_vectors and reference_vectors) using k-Nearest Neighbors algorithm.
        It returns the original query labels, predicted medicine labels, and the indices of the most similar medicines
        in the reference set.

        :param q_labels: a list of ground truth medicine names
        :param r_labels: a list of reference medicine names
        :param reference_vectors: a list of embedded vectors for the reference set
        :param query_vectors: a list of embedded vectors for the query set
        :param n_neighbors: the number of neighbors to consider in the k-NN algorithm
        :return: the original query labels, predicted medicine labels, and indices of the most similar medicines in the
        reference set
        """

        logging.info("Comparing query and reference vectors using kNN")

        predicted_medicine_knn = []
        corresp_sim_knn = []
        most_similar_indices_knn = []

        # convert PyTorch Tensor to numpy array
        reference_vectors_array = np.array([vec.numpy() for vec in reference_vectors]) \
            if isinstance(reference_vectors[0], torch.Tensor) else np.array(reference_vectors)
        query_vectors_array = np.array([vec.numpy() for vec in query_vectors]) \
            if isinstance(query_vectors[0], torch.Tensor) else np.array(query_vectors)

        # Initialise our KNN model. n_neighbors is manually set to 5. You can change this as needed
        knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        knn_model.fit(reference_vectors_array)

        for idx_query, query_vector in tqdm(enumerate(query_vectors_array),
                                            total=len(query_vectors_array),
                                            desc="Comparing process"):

            # Fetch distance and indices of the nearest neighbours in the reference set
            scores_knn, indices_knn = knn_model.kneighbors([query_vector])

            # Convert the distances to a list for further processing
            similarity_scores_knn_dist = scores_knn.tolist()[0]
            corresp_sim_knn.append(similarity_scores_knn_dist)

            # Store the first index (which is the closest in KNN's n_neighbors) and the associated label
            predicted = indices_knn[0][0]
            predicted_medicine_knn.append(r_labels[predicted])

            # Calculate most similar indices for k-NN
            most_similar_indices_knn.append(predicted)

            # Calculate top-1 accuracy
            if r_labels[predicted] == q_labels[idx_query]:
                self.num_correct_top1 += 1

            # Calculate top-5 accuracy. Take note that in KNN, the closest 5 neighbors are returned sorted in increasing
            # distance order
            top5_predicted_medicines = [r_labels[i] for i in indices_knn[0]]
            if q_labels[idx_query] in top5_predicted_medicines:
                self.num_correct_top5 += 1

            # Find index position of the ground truth medicine
            index = top5_predicted_medicines.index(q_labels[idx_query]) \
                if q_labels[idx_query] in top5_predicted_medicines else -1
            self.top5_indices.append(index)

        self.accuracy_top1 = self.num_correct_top1 / len(query_vectors)
        self.accuracy_top5 = self.num_correct_top5 / len(query_vectors)

        # Calculate confidence
        # Flatten the list of lists to get a single list of all similarity scores
        flat_similarity_scores = [max(sublist) for sublist in corresp_sim_knn]

        # Calculate confidence
        confidence_percentages = [1 - (score / max(flat_similarity_scores)) for score in flat_similarity_scores]
        self.confidence_percentages = [cp * 100 for cp in confidence_percentages]

        return q_labels, predicted_medicine_knn, most_similar_indices_knn

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- D I S P L A Y   R E S U L T S ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def display_results(self, ground_truth_labels, predicted_labels, query_vectors) -> None:
        """
        Display the results of the prediction.

        :param ground_truth_labels: List, ground truth labels for the queries.
        :param predicted_labels: List, predicted labels for the queries.
        :param query_vectors: List, vectors representing the queries.
        :return: None
        """

        # Create dataframe
        df = pd.DataFrame(list(zip(ground_truth_labels, predicted_labels)),
                          columns=['GT Medicine Name', 'Predicted Medicine Name (ED)'])
        df['Confidence Percentage'] = self.confidence_percentages
        df['Position of the correct label in the list'] = self.top5_indices

        df_stat = [
            ["Correctly predicted (Top-1):", f'{self.num_correct_top1}'],
            ["Correctly predicted (Top-5):", f'{self.num_correct_top5}'],
            ["Miss predicted:", f'{len(query_vectors) - self.num_correct_top1}'],
            ['Accuracy (Top-1):', f'{self.accuracy_top1:.4%}'],
            ['Accuracy (Top-5):', f'{self.accuracy_top5:.4%}']
        ]
        df_stat = pd.DataFrame(df_stat, columns=['Metric', 'Value'])

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        logging.info(df)
        logging.info(df_stat)

        df_combined = pd.concat([df, df_stat], ignore_index=True)
        df_combined.to_csv(
            os.path.join(self.main_network_config.get('prediction_folder').get(self.cfg.dataset_type),
                         f"{self.timestamp}_{self.cfg.type_of_loss_func}_stream_network_prediction.txt"),
            sep='\t', index=True
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------- M A I N ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def main(self) -> None:
        """
        Executes the pipeline for prediction.
        :return: None
        """

        # Calculate query vectors
        query_vecs, q_labels, q_images_path = (
            self.get_vectors(
                contour_dir=self.sub_network_config.get("Contour").get("query").get(self.cfg.dataset_type),
                lbp_dir=self.sub_network_config.get("LBP").get("query").get(self.cfg.dataset_type),
                rgb_dir=self.sub_network_config.get("RGB").get("query").get(self.cfg.dataset_type),
                texture_dir=self.sub_network_config.get("Texture").get("query").get(self.cfg.dataset_type),
                operation="query"
            )
        )

        # Calculate reference vectors
        if self.cfg.load_ref_vector:
            latest_ref_vec_pt_file = (
                find_latest_file_in_latest_directory(
                    path=self.main_network_config.get('ref_vectors_folder').get(self.cfg.dataset_type),
                    type_of_loss=self.cfg.type_of_loss_func
                )
            )
            data = torch.load(latest_ref_vec_pt_file)

            ref_vecs = data['vectors']
            r_labels = data['labels']
            r_images_path = data['images_path']
            logging.info("Reference vectors has been loaded!")
        else:
            ref_vecs, r_labels, r_images_path = \
                self.get_vectors(
                    contour_dir=self.sub_network_config.get("Contour").get("ref").get(self.cfg.dataset_type),
                    lbp_dir=self.sub_network_config.get("LBP").get("ref").get(self.cfg.dataset_type),
                    rgb_dir=self.sub_network_config.get("RGB").get("ref").get(self.cfg.dataset_type),
                    texture_dir=self.sub_network_config.get("Texture").get("ref").get(self.cfg.dataset_type),
                    operation="reference")

        gt, pred_ed, indices = self.compare_query_and_reference_vectors(q_labels, r_labels, ref_vecs, query_vecs)
        self.display_results(gt, pred_ed, query_vecs)

        # Plot query and reference medicines
        plot_dir = os.path.join(self.main_network_config.get('plotting_folder').get(self.cfg.dataset_type),
                                f"{self.timestamp}_{self.cfg.type_of_loss_func}")
        plot_ref_query_images(indices, q_images_path, r_images_path, gt, pred_ed, plot_dir)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- __M A I N__ ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        pill_rec = PredictStreamNetwork()
        pill_rec.main()
    except KeyboardInterrupt as kie:
        logging.error(kie)
