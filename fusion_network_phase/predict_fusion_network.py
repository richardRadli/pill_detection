"""
File: predict_fusion_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 18, 2023

Description: This program implements the prediction for fusion networks.
"""

import colorama
import logging
import numpy as np
import os
import pandas as pd
import torch

from torchvision import transforms
from tqdm import tqdm
from typing import List, Tuple
from PIL import Image

from config.config import ConfigFusionNetwork, ConfigStreamNetwork
from config.config_selector import sub_stream_network_configs, fusion_network_config
from fusion_network_models.fusion_network_selector import NetworkFactory
from utils.utils import (use_gpu_if_available, create_timestamp, find_latest_file_in_latest_directory,
                         plot_confusion_matrix, plot_ref_query_images, setup_logger)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++ P R E D I C T   F U S I O N   N E T W O R K ++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PredictFusionNetwork:
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- __I N I T__ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        # Setup logger
        self.top5_indices = []
        self.confidence_percentages = None
        self.accuracy_top5 = None
        self.accuracy_top1 = None
        self.num_correct_top5 = 0
        self.num_correct_top1 = 0
        setup_logger()

        # Load config
        self.cfg_fusion_net = ConfigFusionNetwork().parse()

        # Set up tqdm colours
        colorama.init()

        # Create time stamp
        self.timestamp = create_timestamp()

        # Set up class variables
        self.preprocess_rgb = None
        self.preprocess_con_tex_lbp = None
        self.query_image_tex = None
        self.query_image_rgb = None
        self.query_image_con = None

        # Load networks
        self.fusion_network_config = fusion_network_config(network_type=self.cfg_fusion_net.type_of_net)
        self.cfg_stream_net = ConfigStreamNetwork().parse()
        self.subnetwork_config = sub_stream_network_configs(self.cfg_stream_net)
        self.network_cfg_contour = self.subnetwork_config.get("Contour")
        self.network_cfg_lbp = self.subnetwork_config.get("LBP")
        self.network_cfg_rgb = self.subnetwork_config.get("RGB")
        self.network_cfg_texture = self.subnetwork_config.get("Texture")

        self.network = self.load_networks()

        # Set up transforms
        self.preprocess_rgb = transforms.Compose([transforms.Resize((self.network_cfg_rgb.get("image_size"),
                                                                     self.network_cfg_rgb.get("image_size"))),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.preprocess_con_tex_lbp = \
            transforms.Compose([transforms.Resize((self.network_cfg_contour.get("image_size"),
                                                   self.network_cfg_contour.get("image_size"))),
                                transforms.Grayscale(),
                                transforms.ToTensor()])

        # Select device
        self.device = use_gpu_if_available()

        self.plot_dir = os.path.join(
            self.fusion_network_config.get('plotting_folder').get(self.cfg_stream_net.dataset_type),
            f"{self.timestamp}"
        )
        self.plot_confusion_matrix_dir = os.path.join(
            self.fusion_network_config.get("confusion_matrix").get(self.cfg_stream_net.dataset_type),
            f"{self.timestamp}"
        )

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- L O A D   N E T W O R K S -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def load_networks(self):
        """
        This function finds and loads the latest FusionNetwork.

        :return: <class 'fusion_network.FusionNet'>
        """

        latest_con_pt_file = find_latest_file_in_latest_directory(
            path=self.fusion_network_config.get("weights_folder").get(self.cfg_stream_net.dataset_type),
            type_of_loss=self.cfg_fusion_net.type_of_loss_func
        )
        network_fusion = NetworkFactory.create_network(fusion_network_type=self.cfg_fusion_net.type_of_net,
                                                       type_of_net=self.cfg_stream_net.type_of_net,
                                                       network_cfg_contour=self.network_cfg_contour,
                                                       network_cfg_lbp=self.network_cfg_lbp,
                                                       network_cfg_rgb=self.network_cfg_rgb,
                                                       network_cfg_texture=self.network_cfg_texture)
        network_fusion.load_state_dict(torch.load(latest_con_pt_file))
        network_fusion.contour_network.eval()
        network_fusion.lbp_network.eval()
        network_fusion.rgb_network.eval()
        network_fusion.texture_network.eval()
        network_fusion.eval()

        return network_fusion

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- G E T   V E C T O R S ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_vectors(self, contour_dir: str, lbp_dir: str, rgb_dir: str, texture_dir: str, operation: str) -> \
            Tuple[List, List, List]:
        """
        Get feature vectors for images.

        :param contour_dir: path to the directory containing contour images
        :param rgb_dir: path to the directory containing RGB images
        :param texture_dir: path to the directory containing texture images
        :param lbp_dir: path to the directory containing LBP images
        :param operation: name of the operation being performed
        :return: tuple containing three lists - vectors, labels, and images_path
        """

        medicine_classes = os.listdir(rgb_dir)
        vectors = []
        labels = []
        images_path = []
        color = colorama.Fore.BLUE if operation == "query" else colorama.Fore.RED

        self.network = self.network.to(device=self.device)

        for med_class in tqdm(medicine_classes, desc=color + "Processing classes of the %s images" % operation):
            image_paths_con = os.listdir(os.path.join(contour_dir, med_class))
            image_paths_lbp = os.listdir(os.path.join(lbp_dir, med_class))
            image_paths_rgb = os.listdir(os.path.join(rgb_dir, med_class))
            image_paths_tex = os.listdir(os.path.join(texture_dir, med_class))

            for idx, (con, lbp, rgb, tex) in \
                    enumerate(zip(image_paths_con, image_paths_lbp, image_paths_rgb, image_paths_tex)):
                con_image = Image.open(os.path.join(contour_dir, med_class, con))
                con_image = self.preprocess_con_tex_lbp(con_image)

                lbp_image = Image.open(os.path.join(lbp_dir, med_class, lbp))
                lbp_image = self.preprocess_con_tex_lbp(lbp_image)

                rgb_image = Image.open(os.path.join(rgb_dir, med_class, rgb))
                images_path.append(os.path.join(rgb_dir, med_class, rgb))
                rgb_image = self.preprocess_rgb(rgb_image)

                tex_image = Image.open(os.path.join(texture_dir, med_class, tex))
                tex_image = self.preprocess_con_tex_lbp(tex_image)

                with torch.no_grad():
                    # Move input to GPU
                    con_image = con_image.unsqueeze(0).to(self.device)
                    lbp_image = lbp_image.unsqueeze(0).to(self.device)
                    rgb_image = rgb_image.unsqueeze(0).to(self.device)
                    tex_image = tex_image.unsqueeze(0).to(self.device)

                    vector = self.network(con_image, lbp_image, rgb_image, tex_image).squeeze().cpu()

                vectors.append(vector)
                labels.append(med_class)

            if operation == "reference":
                ref_save_dir = (
                    os.path.join(
                        self.fusion_network_config.get('ref_vectors_folder').get(self.cfg_stream_net.dataset_type),
                        f"{self.timestamp}_{self.cfg_fusion_net.type_of_loss_func}"
                    )
                )
                os.makedirs(ref_save_dir, exist_ok=True)
                torch.save({'vectors': vectors, 'labels': labels, 'images_path': images_path},
                           os.path.join(ref_save_dir, "ref_vectors.pt"))

        return vectors, labels, images_path

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------- M E A S U R E   S I M I L A R I T Y   A N D   D I S T A N C E -------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compare_query_ref_vectors_euc_dist(self, q_labels: list, r_labels: list, reference_vectors: list,
                                           query_vectors: list) -> Tuple[List[str], List[str], List[int]]:
        """
        This method measures the Euclidean distance between two sets of labels (q_labels and r_labels) and their
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

        # Move vectors to GPU
        reference_vectors_tensor = torch.stack([torch.as_tensor(vec).to(self.device) for vec in reference_vectors])
        query_vectors_tensor = torch.stack([torch.as_tensor(vec).to(self.device) for vec in query_vectors])

        for idx_query, query_vector in tqdm(enumerate(query_vectors_tensor), total=len(query_vectors_tensor),
                                            desc=colorama.Fore.WHITE + "Comparing process"):
            scores_euc_dist = torch.norm(query_vector - reference_vectors_tensor, dim=1)

            # Move scores to CPU for further processing
            similarity_scores_euc_dist.append(scores_euc_dist.cpu().tolist())

            # Calculate and store the most similar reference vector, predicted medicine label, and corresponding
            # minimum Euclidean distance for each query vector
            most_similar_indices_euc_dist = [scores.index(min(scores)) for scores in similarity_scores_euc_dist]
            predicted_medicine = r_labels[most_similar_indices_euc_dist[idx_query]]
            predicted_medicine_euc_dist.append(predicted_medicine)

            most_similar_indices_and_scores = [(i, min(scores)) for i, scores in enumerate(similarity_scores_euc_dist)]
            corresp_sim_euc_dist.append(most_similar_indices_and_scores[idx_query][1])

            # Calculate top-1 accuracy
            if predicted_medicine == q_labels[idx_query]:
                self.num_correct_top1 += 1

            # Calculate top-5 accuracy
            top5_predicted_medicines = [r_labels[i] for i in torch.argsort(scores_euc_dist)[:5]]
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

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- D I S P L A Y   R E S U L T S ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def display_results(self, ground_truth_labels, predicted_labels, query_vectors) -> None:
        """

        :param ground_truth_labels:
        :param predicted_labels:
        :param query_vectors:
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
            ["Miss predicted top 1:", f'{len(query_vectors) - self.num_correct_top1}'],
            ["Miss predicted top 5:", f'{len(query_vectors) - self.num_correct_top5}'],
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
            os.path.join(self.fusion_network_config.get('prediction_folder').get(self.cfg_stream_net.dataset_type),
                         f"{self.timestamp}_{self.cfg_fusion_net.type_of_loss_func}_fusion_network_prediction.txt"),
            sep='\t', index=True
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------- M A I N ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def main(self) -> None:
        """

        :return: None
        """

        query_vectors, q_labels, q_images_path = self.get_vectors(
            contour_dir=self.subnetwork_config.get("Contour").get("query").get(self.cfg_stream_net.dataset_type),
            lbp_dir=self.subnetwork_config.get("LBP").get("query").get(self.cfg_stream_net.dataset_type),
            rgb_dir=self.subnetwork_config.get("RGB").get("query").get(self.cfg_stream_net.dataset_type),
            texture_dir=self.subnetwork_config.get("Texture").get("query").get(self.cfg_stream_net.dataset_type),
            operation="query")

        ref_vectors, r_labels, r_images_path = self.get_vectors(
            contour_dir="D:/storage/pill_detection/datasets/ogyei/Reference/stream_images/contour",
            #self.subnetwork_config.get("Contour").get("ref").get(self.cfg_stream_net.dataset_type),
            lbp_dir="D:/storage/pill_detection/datasets/ogyei/Reference/stream_images/lbp",
            #self.subnetwork_config.get("LBP").get("ref").get(self.cfg_stream_net.dataset_type),
            rgb_dir="D:/storage/pill_detection/datasets/ogyei/Reference/stream_images/rgb",
            #self.subnetwork_config.get("RGB").get("ref").get(self.cfg_stream_net.dataset_type),
            texture_dir="D:/storage/pill_detection/datasets/ogyei/Reference/stream_images/texture",
            #self.subnetwork_config.get("Texture").get("ref").get(self.cfg_stream_net.dataset_type),
            operation="reference")

        gt, pred_ed, indices = (
            self.compare_query_ref_vectors_euc_dist(q_labels, r_labels, ref_vectors, query_vectors))

        self.display_results(gt, pred_ed, query_vectors)

        plot_ref_query_images(
            indices=indices,
            q_images_path=q_images_path,
            r_images_path=r_images_path,
            gt=gt,
            pred_ed=pred_ed,
            output_folder=self.plot_dir
        )

        plot_confusion_matrix(
            gt=gt,
            pred=pred_ed,
            out_path=self.plot_confusion_matrix_dir
        )


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- __M A I N__ -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        pfn = PredictFusionNetwork()
        pfn.main()
    except KeyboardInterrupt as kie:
        print(kie)
