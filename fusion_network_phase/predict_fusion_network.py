import os

import pandas as pd
import torch

from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from const import CONST
from config import ConfigStreamNetwork
from fusion_network import FusionNet
from utils.utils import create_timestamp, find_latest_file, plot_ref_query_images

cfg = ConfigStreamNetwork().parse()


class PredictFusionNetwork:
    def __init__(self):
        # Create time stamp
        self.timestamp = create_timestamp()

        self.preprocess_rgb = None
        self.preprocess_con_tex = None
        self.query_image_tex = None
        self.query_image_rgb = None
        self.query_image_con = None

        self.network = self.load_networks()
        self.network.eval()

        self.preprocess_rgb = transforms.Compose([transforms.Resize((cfg.img_size, cfg.img_size)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.preprocess_con_tex = transforms.Compose([transforms.Resize((cfg.img_size, cfg.img_size)),
                                                      transforms.Grayscale(),
                                                      transforms.ToTensor()])

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- L O A D   N E T W O R K S -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def load_networks():
        latest_con_pt_file = find_latest_file(CONST.dir_fusion_net_weights)
        network_fusion = FusionNet()
        network_fusion.load_state_dict(torch.load(latest_con_pt_file))

        return network_fusion

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
                    vector = \
                        self.network(con_image.unsqueeze(0), rgb_image.unsqueeze(0), tex_image.unsqueeze(0)).squeeze()
                vectors.append(vector)
                labels.append(med_class)

        return vectors, labels, images_path

    def measure_similarity_and_distance(self, q_labels, r_labels, reference_vectors, query_vectors):
        """

        :param q_labels:
        :param r_labels:
        :param reference_vectors:
        :param query_vectors:
        :return:
        """

        similarity_scores_euc_dist = []
        predicted_medicine_euc_dist = []
        corresp_sim_euc_dist = []
        most_similar_indices_euc_dist = []
        num_correct = 0

        for idx_query, query_vector in tqdm(enumerate(query_vectors), total=len(query_vectors),
                                            desc="Comparing process"):
            scores_e = []
            for idx_ref, reference_vector in enumerate(reference_vectors):
                score_e = torch.pairwise_distance(query_vector, reference_vector).item()
                scores_e.append(score_e)

            similarity_scores_euc_dist.append(scores_e)

            most_similar_indices_euc_dist = [scores.index(min(scores)) for scores in similarity_scores_euc_dist]
            predicted_medicine = r_labels[most_similar_indices_euc_dist[idx_query]]
            predicted_medicine_euc_dist.append(predicted_medicine)

            most_similar_indices_and_scores_e = [(i, min(scores)) for i, scores in
                                                 enumerate(similarity_scores_euc_dist)]
            corresp_sim_euc_dist.append(most_similar_indices_and_scores_e[idx_query][1])

            if predicted_medicine == q_labels[idx_query]:
                num_correct += 1

        accuracy = num_correct / len(query_vectors)

        df = pd.DataFrame(list(zip(q_labels, predicted_medicine_euc_dist)),
                          columns=['GT Medicine Name', 'Predicted Medicine Name (ED)'])
        df.loc[len(df)] = ["Correctly predicted:", f'{num_correct}']
        df.loc[len(df)] = ["Miss predicted:", f'{len(query_vectors) - num_correct}']
        df.loc[len(df)] = ['Accuracy:', f'{accuracy:.4%}']
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print(df)

        df.to_csv(os.path.join(CONST.dir_fusion_network_predictions, self.timestamp + "_fusion_network_prediction.txt"),
                  sep='\t', index=True)

        return q_labels, predicted_medicine_euc_dist, most_similar_indices_euc_dist

    def main(self):
        query_vecs, q_labels, q_images_path = self.get_vectors(contour_dir=CONST.dir_query_contour,
                                                               rgb_dir=CONST.dir_query_rgb,
                                                               texture_dir=CONST.dir_query_texture,
                                                               operation="query")

        ref_vecs, r_labels, r_images_path = self.get_vectors(contour_dir=CONST.dir_contour,
                                                             rgb_dir=CONST.dir_rgb,
                                                             texture_dir=CONST.dir_texture,
                                                             operation="reference")

        gt, pred_ed, indices = self.measure_similarity_and_distance(q_labels, r_labels, ref_vecs, query_vecs)
        plot_ref_query_images(indices, q_images_path, r_images_path, gt, pred_ed, operation="fuse")


if __name__ == "__main__":
    try:
        pfn = PredictFusionNetwork()
        pfn.main()
    except KeyboardInterrupt as kie:
        print(kie)
