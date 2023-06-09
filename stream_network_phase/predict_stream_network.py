"""
File: predict_stream_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description:
"""

import logging
import numpy as np
import os
import pandas as pd
import torch

from glob import glob
from torchvision import transforms
from tqdm import tqdm
from typing import List, Tuple, Dict
from PIL import Image

from config.const import DATA_PATH, IMAGES_PATH
from config.config import ConfigStreamNetwork
from config.logger_setup import setup_logger
from network_selector import NetworkFactory
from utils.utils import create_timestamp, find_latest_file_in_latest_directory, \
    plot_ref_query_images, use_gpu_if_available


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

        # Create time stamp
        self.timestamp = create_timestamp()

        self.preprocess_rgb = None
        self.preprocess_con_tex = None

        # Load configs
        self.main_network_config = self.get_main_network_config()
        self.sub_network_config = self.get_subnetwork_config()

        # Load networks
        self.network_con, self.network_rgb, self.network_tex, self.network_lbp = self.load_networks()
        self.network_con.eval()
        self.network_rgb.eval()
        self.network_tex.eval()
        self.network_lbp.eval()

        # Preprocess images
        self.preprocess_rgb = transforms.Compose([transforms.Resize((self.cfg.img_size, self.cfg.img_size)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.preprocess_con_tex = transforms.Compose([transforms.Resize((self.cfg.img_size, self.cfg.img_size)),
                                                      transforms.Grayscale(),
                                                      transforms.ToTensor()])

        # Select device
        self.device = use_gpu_if_available()

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------- G E T   M A I N   N E T W O R K   C O N F I G ---------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_main_network_config(self) -> Dict:
        """
        Returns a dictionary containing the prediction and plotting folder paths for different types of networks
        based on the type_of_net parameter in self.cfg.
        :return: Dictionary containing the prediction and plotting folder paths.
        """

        network_type = self.cfg.type_of_net
        logging.info(network_type)
        network_configs = {
            'StreamNetwork': {
                'prediction_folder': DATA_PATH.get_data_path("predictions_stream_network"),
                'plotting_folder': IMAGES_PATH.get_data_path("plotting_stream_network"),
                'ref_vectors_folder': DATA_PATH.get_data_path("reference_vectors_stream_network")
            },
            'EfficientNet': {
                'prediction_folder': DATA_PATH.get_data_path("predictions_efficient_net"),
                'plotting_folder': IMAGES_PATH.get_data_path("plotting_efficient_net"),
                'ref_vectors_folder': DATA_PATH.get_data_path("reference_vectors_efficient_net")
            },
            'EfficientNetSelfAttention': {
                'prediction_folder': DATA_PATH.get_data_path("predictions_efficient_self_attention_net"),
                'plotting_folder': IMAGES_PATH.get_data_path("plotting_efficient_net_self_attention"),
                'ref_vectors_folder': DATA_PATH.get_data_path("reference_vectors_efficient_net_self_attention")
            }
        }
        if network_type not in network_configs:
            raise ValueError(f'Invalid network type: {network_type}')

        return network_configs[network_type]

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------- G E T   S U B   N E T W O R K   C O N F I G ----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_subnetwork_config(self) -> Dict:
        """
        Returns the configuration of subnetworks

        :return: dictionary containing subnetwork configuration
        """

        network_config = {
            "Contour": {
                "channels": [1, 32, 48, 64, 128, 192, 256],
                "model_weights_dir": {
                    "StreamNetwork": DATA_PATH.get_data_path("weights_stream_network_contour"),
                    "EfficientNet": DATA_PATH.get_data_path("weights_efficient_net_contour"),
                    "EfficientNetSelfAttention":
                        DATA_PATH.get_data_path("weights_efficient_net_self_attention_contour"),
                }.get(self.cfg.type_of_net, DATA_PATH.get_data_path("weights_stream_network_contour")),
                "grayscale": True
            },

            "LBP": {
                "channels": [1, 32, 48, 64, 128, 192, 256],
                "model_weights_dir": {
                    "StreamNetwork": DATA_PATH.get_data_path("weights_stream_network_lbp"),
                    "EfficientNet": DATA_PATH.get_data_path("weights_efficient_net_lbp"),
                    "EfficientNetSelfAttention": DATA_PATH.get_data_path("weights_efficient_net_self_attention_lbp"),
                }.get(self.cfg.type_of_net, DATA_PATH.get_data_path("weights_stream_network_lbp")),
                "grayscale": True
            },

            "RGB": {
                "channels": [3, 64, 96, 128, 256, 384, 512],
                "model_weights_dir": {
                    "StreamNetwork": DATA_PATH.get_data_path("weights_stream_network_rgb"),
                    "EfficientNet": DATA_PATH.get_data_path("weights_efficient_net_rgb"),
                    "EfficientNetSelfAttention": DATA_PATH.get_data_path("weights_efficient_net_self_attention_rgb"),
                }.get(self.cfg.type_of_net, DATA_PATH.get_data_path("weights_stream_network_rgb")),
                "grayscale": False
            },

            "Texture": {
                "channels": [1, 32, 48, 64, 128, 192, 256],
                "model_weights_dir": {
                    "StreamNetwork": DATA_PATH.get_data_path("weights_stream_network_texture"),
                    "EfficientNet": DATA_PATH.get_data_path("weights_efficient_net_texture"),
                    "EfficientNetSelfAttention":
                        DATA_PATH.get_data_path("weights_efficient_net_self_attention_texture"),
                }.get(self.cfg.type_of_net, DATA_PATH.get_data_path("weights_stream_network_texture")),
                "grayscale": True
            },
        }

        return network_config

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- L O A D   N E T W O R K S -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def load_networks(self):
        """
        This function loads the pretrained networks, with the latest .pt files

        :return: The contour, rgb, and texture networks.
        """

        con_config = self.sub_network_config.get("Contour")
        rgb_config = self.sub_network_config.get("RGB")
        tex_config = self.sub_network_config.get("Texture")
        lbp_config = self.sub_network_config.get("LBP")

        latest_con_pt_file = find_latest_file_in_latest_directory(con_config.get("model_weights_dir"))
        latest_rgb_pt_file = find_latest_file_in_latest_directory(rgb_config.get("model_weights_dir"))
        latest_tex_pt_file = find_latest_file_in_latest_directory(tex_config.get("model_weights_dir"))
        latest_lbp_pt_file = find_latest_file_in_latest_directory(lbp_config.get("model_weights_dir"))

        network_con = NetworkFactory.create_network(self.cfg.type_of_net, con_config)
        network_rgb = NetworkFactory.create_network(self.cfg.type_of_net, rgb_config)
        network_tex = NetworkFactory.create_network(self.cfg.type_of_net, tex_config)
        network_lbp = NetworkFactory.create_network(self.cfg.type_of_net, lbp_config)

        network_con.load_state_dict(torch.load(latest_con_pt_file))
        network_rgb.load_state_dict(torch.load(latest_rgb_pt_file))
        network_tex.load_state_dict(torch.load(latest_tex_pt_file))
        network_lbp.load_state_dict(torch.load(latest_lbp_pt_file))

        return network_con, network_rgb, network_tex, network_lbp

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

        medicine_classes = os.listdir(rgb_dir)
        vectors = []
        labels = []
        images_path = []

        # Move the model to the GPU
        self.network_con = self.network_con.to(self.device)
        self.network_lbp = self.network_lbp.to(self.device)
        self.network_rgb = self.network_rgb.to(self.device)
        self.network_tex = self.network_tex.to(self.device)

        for image_name in tqdm(medicine_classes, desc="\nProcess %s images" % operation):
            # Collecting images
            image_paths_con = os.listdir(os.path.join(contour_dir, image_name))
            image_paths_rgb = os.listdir(os.path.join(rgb_dir, image_name))
            image_paths_tex = os.listdir(os.path.join(texture_dir, image_name))
            image_paths_lbp = os.listdir(os.path.join(lbp_dir, image_name))

            for idx, (con, rgb, tex, lbp) in enumerate(zip(
                    image_paths_con, image_paths_rgb, image_paths_tex, image_paths_lbp)):
                # Open images and convert them to tensors
                con_image = Image.open(os.path.join(contour_dir, image_name, con))
                con_image = self.preprocess_con_tex(con_image)

                lbp_image = Image.open(os.path.join(lbp_dir, image_name, lbp))
                lbp_image = self.preprocess_con_tex(lbp_image)

                rgb_image = Image.open(os.path.join(rgb_dir, image_name, rgb))
                images_path.append(os.path.join(rgb_dir, image_name, rgb))
                rgb_image = self.preprocess_rgb(rgb_image)

                tex_image = Image.open(os.path.join(texture_dir, image_name, tex))
                tex_image = self.preprocess_con_tex(tex_image)

                # Make prediction
                with torch.no_grad():
                    # Move input to GPU
                    con_image = con_image.unsqueeze(0).to(self.device)
                    lbp_image = lbp_image.unsqueeze(0).to(self.device)
                    rgb_image = rgb_image.unsqueeze(0).to(self.device)
                    tex_image = tex_image.unsqueeze(0).to(self.device)

                    # Perform computation on GPU and move result back to CPU
                    vector1 = self.network_con(con_image).squeeze().cpu()
                    vector2 = self.network_lbp(lbp_image).squeeze().cpu()
                    vector3 = self.network_rgb(rgb_image).squeeze().cpu()
                    vector4 = self.network_tex(tex_image).squeeze().cpu()

                vector = torch.cat((vector1, vector2, vector3, vector4), dim=0)
                vectors.append(vector)
                labels.append(image_name)

        if operation == "reference":
            torch.save({'vectors': vectors, 'labels': labels, 'images_path': images_path},
                       os.path.join(self.main_network_config['ref_vectors_folder'], self.timestamp + "_ref_vectors.pt"))

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
        reference set, and Mean Average Precision (mAP).

        :param q_labels: a list of ground truth medicine names
        :param r_labels: a list of reference medicine names
        :param reference_vectors: a numpy array of embedded vectors for the reference set
        :param query_vectors: a numpy array of embedded vectors for the query set
        :return: the original query labels, predicted medicine labels, indices of the most similar medicines in the
        reference set, and mAP
        """

        similarity_scores_euc_dist = []
        predicted_medicine_euc_dist = []
        corresp_sim_euc_dist = []
        most_similar_indices_euc_dist = []
        num_correct_top1 = 0
        num_correct_top5 = 0

        # Move vectors to GPU
        reference_vectors_tensor = torch.stack([torch.as_tensor(vec).to(self.device) for vec in reference_vectors])
        query_vectors_tensor = torch.stack([torch.as_tensor(vec).to(self.device) for vec in query_vectors])

        for idx_query, query_vector in tqdm(enumerate(query_vectors_tensor), total=len(query_vectors_tensor),
                                            desc="Comparing process"):
            scores_e = torch.norm(query_vector - reference_vectors_tensor, dim=1)

            # Move scores to CPU for further processing
            similarity_scores_euc_dist.append(scores_e.cpu().tolist())

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
                num_correct_top1 += 1

            # Calculate top-5 accuracy
            top5_predicted_medicines = [r_labels[i] for i in torch.argsort(scores_e)[:5]]
            if q_labels[idx_query] in top5_predicted_medicines:
                num_correct_top5 += 1

        accuracy_top1 = num_correct_top1 / len(query_vectors)
        accuracy_top5 = num_correct_top5 / len(query_vectors)

        # Calculate confidence
        confidence_percentages = [1 - (score / max(scores)) for score, scores in
                                  zip(corresp_sim_euc_dist, similarity_scores_euc_dist)]
        confidence_percentages = [cp * 100 for cp in confidence_percentages]

        # Find index position of the ground truth medicine
        top5_indices = []
        for idx_query, query_label in enumerate(q_labels):
            top5_predicted_medicines = [r_labels[i] for i in np.argsort(similarity_scores_euc_dist[idx_query])[:5]]
            if query_label in top5_predicted_medicines:
                index = top5_predicted_medicines.index(query_label)
            else:
                index = -1
            top5_indices.append(index)

        # Create dataframe
        df = pd.DataFrame(list(zip(q_labels, predicted_medicine_euc_dist)),
                          columns=['GT Medicine Name', 'Predicted Medicine Name (ED)'])
        df['Confidence Percentage'] = confidence_percentages
        df['Position of the correct label in the list'] = top5_indices

        df_stat = [
            ["Correctly predicted (Top-1):", f'{num_correct_top1}'],
            ["Correctly predicted (Top-5):", f'{num_correct_top5}'],
            ["Miss predicted:", f'{len(query_vectors) - num_correct_top1}'],
            ['Accuracy (Top-1):', f'{accuracy_top1:.4%}'],
            ['Accuracy (Top-5):', f'{accuracy_top5:.4%}']
        ]
        df_stat = pd.DataFrame(df_stat, columns=['Metric', 'Value'])

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        logging.info(df)
        logging.info(df_stat)

        df_combined = pd.concat([df, df_stat], ignore_index=True)

        df_combined.to_csv(os.path.join(self.main_network_config['prediction_folder'],
                                        self.timestamp + "_stream_network_prediction.txt"), sep='\t', index=True)

        return q_labels, predicted_medicine_euc_dist, most_similar_indices_euc_dist

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------- M A I N ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def main(self) -> None:
        """
        Executes the pipeline for prediction.
        :return: None
        """

        # Calculate query vectors
        query_vecs, q_labels, q_images_path = self.get_vectors(contour_dir=IMAGES_PATH.get_data_path("query_contour"),
                                                               lbp_dir=IMAGES_PATH.get_data_path("query_lbp"),
                                                               rgb_dir=IMAGES_PATH.get_data_path("query_rgb"),
                                                               texture_dir=IMAGES_PATH.get_data_path("query_texture"),
                                                               operation="query")

        # Calculate reference vectors
        if self.cfg.load_ref_vector:
            latest_ref_vec_pt_file = \
                max(glob(os.path.join(self.main_network_config['ref_vectors_folder'], "*.pt")), key=os.path.getctime)
            data = torch.load(latest_ref_vec_pt_file)

            ref_vecs = data['vectors']
            r_labels = data['labels']
            r_images_path = data['images_path']
            logging.info("Reference vectors has been loaded!")
        else:
            ref_vecs, r_labels, r_images_path = \
                self.get_vectors(contour_dir=IMAGES_PATH.get_data_path("ref_train_contour"),
                                 lbp_dir=IMAGES_PATH.get_data_path("ref_train_lbp"),
                                 rgb_dir=IMAGES_PATH.get_data_path("ref_train_rgb"),
                                 texture_dir=IMAGES_PATH.get_data_path("ref_train_texture"),
                                 operation="reference")

        # Compare query and reference vectors
        gt, pred_ed, indices = self.compare_query_and_reference_vectors(q_labels, r_labels, ref_vecs, query_vecs)

        # Plot query and reference medicines
        plot_ref_query_images(indices, q_images_path, r_images_path, gt, pred_ed,
                              self.main_network_config['plotting_folder'])


if __name__ == "__main__":
    try:
        pill_rec = PredictStreamNetwork()
        pill_rec.main()
    except KeyboardInterrupt as kie:
        logging.error(kie)
