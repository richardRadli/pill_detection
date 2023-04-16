import os

import pandas as pd
import torch

from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from const import CONST
from config import ConfigStreamNetwork
from stream_network import StreamNetwork
from utils.utils import find_latest_file, plot_ref_query_images

cfg = ConfigStreamNetwork().parse()


class PillRecognition:
    def __init__(self):
        self.preprocess_rgb = None
        self.preprocess_con_tex = None
        self.query_image_tex = None
        self.query_image_rgb = None
        self.query_image_con = None

        self.network_con, self.network_rgb, self.network_tex = self.load_networks()

        self.network_con.eval()
        self.network_rgb.eval()
        self.network_tex.eval()

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
        """
        This function loads the pretrained networks, with the latest .pt files
        :return: The contour, rgb, and texture networks.
        """

        list_of_channels_tex_con = [1, 32, 48, 64, 128, 192, 256]
        list_of_channels_rgb = [3, 64, 96, 128, 256, 384, 512]

        latest_con_pt_file = find_latest_file(CONST.dir_stream_contour_model_weights)
        latest_rgb_pt_file = find_latest_file(CONST.dir_stream_rgb_model_weights)
        latest_tex_pt_file = find_latest_file(CONST.dir_stream_texture_model_weights)

        network_con = StreamNetwork(loc=list_of_channels_tex_con)
        network_rgb = StreamNetwork(loc=list_of_channels_rgb)
        network_tex = StreamNetwork(loc=list_of_channels_tex_con)

        network_con.load_state_dict(torch.load(latest_con_pt_file))
        network_rgb.load_state_dict(torch.load(latest_rgb_pt_file))
        network_tex.load_state_dict(torch.load(latest_tex_pt_file))

        return network_con, network_rgb, network_tex

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- G E T   V E C T O R S ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_vectors(self, contour_dir, rgb_dir, texture_dir, operation):
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
                    vector1 = self.network_con(con_image.unsqueeze(0)).squeeze()
                    vector2 = self.network_rgb(rgb_image.unsqueeze(0)).squeeze()
                    vector3 = self.network_tex(tex_image.unsqueeze(0)).squeeze()
                vector = torch.cat((vector1, vector2, vector3), dim=0)
                vectors.append(vector)
                labels.append(med_class)

        return vectors, labels, images_path

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------- M E A S U R E   C O S S I M   A N D   E U C D I S T ------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def measure_similarity_and_distance(q_labels, r_labels, reference_vectors, query_vectors):
        """

        :param q_labels:
        :param r_labels:
        :param reference_vectors:
        :param query_vectors:
        :return:
        """

        similarity_scores_cos_sim = []
        similarity_scores_euc_dist = []
        predicted_medicine_cos_sim = []
        corresp_sim_cos_sim = []
        predicted_medicine_euc_dist = []
        corresp_sim_euc_dist = []
        most_similar_indices_cos_sim = []

        for idx_query, query_vector in tqdm(enumerate(query_vectors), total=len(query_vectors),
                                            desc="Comparing process"):
            scores = []
            scores_e = []
            for idx_ref, reference_vector in enumerate(reference_vectors):
                score = torch.nn.functional.cosine_similarity(query_vector, reference_vector, dim=0).item()
                scores.append(score)

                score_e = torch.pairwise_distance(query_vector, reference_vector).item()
                scores_e.append(score_e)

            similarity_scores_cos_sim.append(scores)
            similarity_scores_euc_dist.append(scores_e)

            most_similar_indices_cos_sim = [scores.index(max(scores)) for scores in similarity_scores_cos_sim]
            predicted_medicine_cos_sim.append(r_labels[most_similar_indices_cos_sim[idx_query]])

            most_similar_indices_and_scores = [(i, max(scores)) for i, scores in enumerate(similarity_scores_cos_sim)]
            corresp_sim_cos_sim.append(most_similar_indices_and_scores[idx_query][1])

            most_similar_indices_euc_dist = [scores.index(min(scores)) for scores in similarity_scores_euc_dist]
            predicted_medicine_euc_dist.append(r_labels[most_similar_indices_euc_dist[idx_query]])

            most_similar_indices_and_scores_e = [(i, min(scores)) for i, scores in
                                                 enumerate(similarity_scores_euc_dist)]
            corresp_sim_euc_dist.append(most_similar_indices_and_scores_e[idx_query][1])

        df = pd.DataFrame(list(zip(q_labels, predicted_medicine_cos_sim, predicted_medicine_euc_dist)),
                          columns=['GT Medicine Name', 'Predicted Medicine Name (CS)', 'Predicted Medicine Name (ED)'])

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print(df)

        return q_labels, predicted_medicine_cos_sim, predicted_medicine_euc_dist, most_similar_indices_cos_sim

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- M E A S U R E   A C C U R A C Y ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def measure_accuracy(gt, pred, op):
        """

        :param gt:
        :param pred:
        :param op:
        :return:
        """

        count = 0
        for i in range(len(gt)):
            if gt[i] == pred[i] or gt[i] == pred[i]:
                count += 1
        print("Accuracy {}: {:.2%}".format(op, count / len(gt)))

    @staticmethod
    def measure_accuracy_comb(gt, pred_cs, pred_ed):
        """

        :param gt:
        :param pred_cs:
        :param pred_ed:
        :return:
        """

        count = 0
        for i in range(len(gt)):
            if gt[i] == pred_cs[i] or gt[i] == pred_ed[i]:
                count += 1
        print("Accuracy [Union of CS and ED]: {:.2%}".format(count / len(gt)))

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------- M A I N ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def main(self):
        """

        :return:
        """

        query_vecs, q_labels, q_images_path = self.get_vectors(contour_dir=CONST.dir_query_contour,
                                                               rgb_dir=CONST.dir_query_rgb,
                                                               texture_dir=CONST.dir_query_texture,
                                                               operation="query")

        ref_vecs, r_labels, r_images_path = self.get_vectors(contour_dir=CONST.dir_contour,
                                                             rgb_dir=CONST.dir_bounding_box,
                                                             texture_dir=CONST.dir_texture,
                                                             operation="reference")

        gt, pred_cs, pred_ed, indecies = self.measure_similarity_and_distance(q_labels, r_labels, ref_vecs, query_vecs)

        self.measure_accuracy(gt, pred_cs, "cos. sim.")
        self.measure_accuracy(gt, pred_ed, "euc. dist.")
        self.measure_accuracy_comb(gt, pred_cs, pred_ed)

        plot_ref_query_images(indecies, q_images_path, r_images_path, gt, pred_ed)


if __name__ == "__main__":
    try:
        pill_rec = PillRecognition()
        pill_rec.main()
    except KeyboardInterrupt as kie:
        print(kie)
