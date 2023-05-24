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

from config.const import CONST
from config.config import ConfigStreamNetwork
from config.logger_setup import setup_logger
from network_selector import NetworkFactory
from utils.utils import use_gpu_if_available, create_timestamp, find_latest_file_in_latest_directory, \
    plot_ref_query_images


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
        self.query_image_tex = None
        self.query_image_rgb = None
        self.query_image_con = None

        # Load configs
        self.main_network_config = self.get_main_network_config()
        self.sub_network_config = self.get_subnetwork_config()

        # Load networks
        self.network_con, self.network_rgb, self.network_tex = self.load_networks()
        self.network_con.eval()
        self.network_rgb.eval()
        self.network_tex.eval()

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
                'prediction_folder': CONST.dir_stream_network_predictions,
                'plotting_folder': CONST.dir_stream_network_pred,
                'ref_vectors_folder': CONST.dir_ref_vectors_stream_net
            },
            'EfficientNet': {
                'prediction_folder': CONST.dir_efficient_net_predictions,
                'plotting_folder': CONST.dir_efficient_net_prediction,
                'ref_vectors_folder': CONST.dir_ref_vectors_efficient_net
            },
            'EfficientNetSelfAttention': {
                'prediction_folder': CONST.dir_efficient_net_self_attention_predictions,
                'plotting_folder': CONST.dir_efficient_net_self_attention_prediction,
                'ref_vectors_folder': CONST.dir_ref_vectors_efficient_net_self_attention
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
            "RGB": {
                "channels": [3, 64, 96, 128, 256, 384, 512],
                "model_weights_dir": {
                    "StreamNetwork": CONST.dir_stream_rgb_model_weights,
                    "EfficientNet": CONST.dir_efficient_net_rgb_model_weights,
                    "EfficientNetSelfAttention": CONST.dir_efficient_net_self_attention_rgb_model_weights
                }.get(self.cfg.type_of_net, CONST.dir_stream_rgb_model_weights),
                "grayscale": False
            },

            "Texture": {
                "channels": [1, 32, 48, 64, 128, 192, 256],
                "model_weights_dir": {
                    "StreamNetwork": CONST.dir_stream_texture_model_weights,
                    "EfficientNet": CONST.dir_efficient_net_texture_model_weights,
                    "EfficientNetSelfAttention": CONST.dir_efficient_net_self_attention_texture_model_weights
                }.get(self.cfg.type_of_net, CONST.dir_stream_texture_model_weights),
                "grayscale": True
            },

            "Contour": {
                "channels": [1, 32, 48, 64, 128, 192, 256],
                "model_weights_dir": {
                    "StreamNetwork": CONST.dir_stream_contour_model_weights,
                    "EfficientNet": CONST.dir_efficient_net_contour_model_weights,
                    "EfficientNetSelfAttention": CONST.dir_efficient_net_self_attention_contour_model_weights
                }.get(self.cfg.type_of_net, CONST.dir_stream_contour_model_weights),
                "grayscale": True
            }
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

        latest_con_pt_file = find_latest_file_in_latest_directory(con_config.get("model_weights_dir"))
        latest_rgb_pt_file = find_latest_file_in_latest_directory(rgb_config.get("model_weights_dir"))
        latest_tex_pt_file = find_latest_file_in_latest_directory(tex_config.get("model_weights_dir"))

        network_con = NetworkFactory.create_network(self.cfg.type_of_net, con_config)
        network_rgb = NetworkFactory.create_network(self.cfg.type_of_net, rgb_config)
        network_tex = NetworkFactory.create_network(self.cfg.type_of_net, tex_config)

        network_con.load_state_dict(torch.load(latest_con_pt_file))
        network_rgb.load_state_dict(torch.load(latest_rgb_pt_file))
        network_tex.load_state_dict(torch.load(latest_tex_pt_file))

        return network_con, network_rgb, network_tex

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- G E T   V E C T O R S ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_vectors(self, contour_dir: str, rgb_dir: str, texture_dir: str, operation: str):
        """
        :param contour_dir: path to the directory containing contour images
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
        self.network_rgb = self.network_rgb.to(self.device)
        self.network_tex = self.network_tex.to(self.device)

        for med_class in tqdm(medicine_classes, desc="\nProcess %s images" % operation):
            # Collecting the images
            image_paths_con = os.listdir(os.path.join(contour_dir, med_class))
            image_paths_rgb = os.listdir(os.path.join(rgb_dir, med_class))
            image_paths_tex = os.listdir(os.path.join(texture_dir, med_class))

            for idx, (con, rgb, tex) in enumerate(zip(image_paths_con, image_paths_rgb, image_paths_tex)):
                # Open images and convert them to tensors
                con_image = Image.open(os.path.join(contour_dir, med_class, con))
                con_image = self.preprocess_con_tex(con_image)

                rgb_image = Image.open(os.path.join(rgb_dir, med_class, rgb))
                images_path.append(os.path.join(rgb_dir, med_class, rgb))
                rgb_image = self.preprocess_rgb(rgb_image)

                tex_image = Image.open(os.path.join(texture_dir, med_class, tex))
                tex_image = self.preprocess_con_tex(tex_image)

                # Make prediction
                with torch.no_grad():
                    # Move input to GPU
                    con_image = con_image.unsqueeze(0).to(self.device)
                    rgb_image = rgb_image.unsqueeze(0).to(self.device)
                    tex_image = tex_image.unsqueeze(0).to(self.device)

                    # Perform computation on GPU and move result back to CPU
                    vector1 = self.network_con(con_image).squeeze().cpu()
                    vector2 = self.network_rgb(rgb_image).squeeze().cpu()
                    vector3 = self.network_tex(tex_image).squeeze().cpu()

                vector = torch.cat((vector1, vector2, vector3), dim=0)
                vectors.append(vector)
                labels.append(med_class)

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
        It returns the original query labels, predicted medicine labels, and the indices of the most similar medicines
        in the reference set.

        :param q_labels: a list of ground truth medicine names
        :param r_labels: a list of reference medicine names
        :param reference_vectors: a numpy array of embedded vectors for the reference set
        :param query_vectors: a numpy array of embedded vectors for the query set
        :return: the original query labels, predicted medicine labels, and indices of the most similar medicines in the
        reference set
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
    def main(self):
        """

        :return:
        """

        # Calculate query vectors
        query_vecs, q_labels, q_images_path = self.get_vectors(contour_dir=CONST.dir_query_contour,
                                                               rgb_dir=CONST.dir_query_rgb,
                                                               texture_dir=CONST.dir_query_texture,
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
            ref_vecs, r_labels, r_images_path = self.get_vectors(contour_dir=CONST.dir_contour,
                                                                 rgb_dir=CONST.dir_rgb,
                                                                 texture_dir=CONST.dir_texture,
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
