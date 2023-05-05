import numpy as np
import os
import pandas as pd
import torch

from torchvision import transforms
from tqdm import tqdm
from typing import List, Tuple
from PIL import Image

from const import CONST
from config import ConfigFusionNetwork
from fusion_network import FusionNet
from utils.utils import use_gpu_if_available, create_timestamp, find_latest_file_in_latest_directory, \
    plot_ref_query_images, find_latest_file_in_directory, prediction_statistics


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++ P R E D I C T   F U S I O N   N E T W O R K ++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PredictFusionNetwork:
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- __I N I T__ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        # Load config
        self.cfg = ConfigFusionNetwork().parse()

        # Create time stamp
        self.timestamp = create_timestamp()

        self.preprocess_rgb = None
        self.preprocess_con_tex = None
        self.query_image_tex = None
        self.query_image_rgb = None
        self.query_image_con = None

        self.network = self.load_networks()
        self.network.eval()

        self.preprocess_rgb = transforms.Compose([transforms.Resize((self.cfg.img_size, self.cfg.img_size)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.preprocess_con_tex = transforms.Compose([transforms.Resize((self.cfg.img_size, self.cfg.img_size)),
                                                      transforms.Grayscale(),
                                                      transforms.ToTensor()])

        self.device = use_gpu_if_available()

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- L O A D   N E T W O R K S -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def load_networks():
        """
        This function finds and loads the latest FusionNetwork.

        :return: <class 'fusion_network.FusionNet'>
        """

        latest_con_pt_file = find_latest_file_in_latest_directory(CONST.dir_fusion_net_weights)
        network_fusion = FusionNet()
        network_fusion.load_state_dict(torch.load(latest_con_pt_file))

        return network_fusion

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- G E T   V E C T O R S ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_vectors(self, contour_dir: str, rgb_dir: str, texture_dir: str, operation: str) -> Tuple[List, List, List]:
        """
        Get feature vectors for images.

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

        self.network = self.network.to(device=self.device)

        for med_class in tqdm(medicine_classes, desc="Process %s images" % operation):
            image_paths_con = os.listdir(os.path.join(contour_dir, med_class))
            image_paths_rgb = os.listdir(os.path.join(rgb_dir, med_class))
            image_paths_tex = os.listdir(os.path.join(texture_dir, med_class))

            for idx, (con, rgb, tex) in enumerate(zip(image_paths_con, image_paths_rgb, image_paths_tex)):
                con_image = Image.open(os.path.join(contour_dir, med_class, con))
                con_image = self.preprocess_con_tex(con_image)

                rgb_image = Image.open(os.path.join(rgb_dir, med_class, rgb))
                images_path.append(os.path.join(rgb_dir, med_class, rgb))
                rgb_image = self.preprocess_rgb(rgb_image)

                tex_image = Image.open(os.path.join(texture_dir, med_class, tex))
                tex_image = self.preprocess_con_tex(tex_image)

                with torch.no_grad():
                    # Move input to GPU
                    con_image = con_image.unsqueeze(0).to(self.device)
                    rgb_image = rgb_image.unsqueeze(0).to(self.device)
                    tex_image = tex_image.unsqueeze(0).to(self.device)

                    vector = \
                        self.network(con_image, rgb_image, tex_image).squeeze().cpu()

                vectors.append(vector)
                labels.append(med_class)

        return vectors, labels, images_path

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------- M E A S U R E   S I M I L A R I T Y   A N D   D I S T A N C E -------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def measure_similarity_and_distance(self, q_labels: list, r_labels: list, reference_vectors: list,
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

        print(df)
        print(df_stat)

        df_combined = pd.concat([df, df_stat], ignore_index=True)

        df_combined.to_csv(
            os.path.join(CONST.dir_fusion_network_predictions, self.timestamp + "_fusion_network_prediction.txt"),
            sep='\t', index=True)

        return q_labels, predicted_medicine_euc_dist, most_similar_indices_euc_dist

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------- M A I N ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def main(self) -> None:
        """

        :return: None
        """

        query_vectors, q_labels, q_images_path = self.get_vectors(contour_dir=CONST.dir_query_contour,
                                                                  rgb_dir=CONST.dir_query_rgb,
                                                                  texture_dir=CONST.dir_query_texture,
                                                                  operation="query")

        ref_vectors, r_labels, r_images_path = self.get_vectors(contour_dir=CONST.dir_contour,
                                                                rgb_dir=CONST.dir_rgb,
                                                                texture_dir=CONST.dir_texture,
                                                                operation="reference")

        gt, pred_ed, indices = self.measure_similarity_and_distance(q_labels, r_labels, ref_vectors, query_vectors)

        stream_net_pred = find_latest_file_in_directory(CONST.dir_stream_network_predictions, "txt")
        fusion_net_pred = find_latest_file_in_directory(CONST.dir_fusion_network_predictions, "txt")
        prediction_statistics(stream_net_pred, fusion_net_pred)

        plot_ref_query_images(indices, q_images_path, r_images_path, gt, pred_ed, operation="fuse")


if __name__ == "__main__":
    try:
        pfn = PredictFusionNetwork()
        pfn.main()
    except KeyboardInterrupt as kie:
        print(kie)