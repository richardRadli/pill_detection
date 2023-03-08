import os

import pandas as pd
import torch
import torch.nn.functional as func

from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from const import CONST
from config import ConfigStreamNetwork
from stream_network import StreamNetwork
from utils.utils import find_latest_file, segment_pills

cfg = ConfigStreamNetwork().parse()


class PillRecognition:
    def __init__(self):
        self.preprocess_rgb = None
        self.preprocess_con_tex = None
        self.query_image_tex = None
        self.query_image_rgb = None
        self.query_image_con = None
        self.network_con, self.network_rgb, self.network_tex = self.load_networks()

        self.preprocess_rgb = transforms.Compose([transforms.Resize((cfg.img_size, cfg.img_size)),
                                                  transforms.ToTensor()])

        self.preprocess_con_tex = transforms.Compose([transforms.Resize((cfg.img_size, cfg.img_size)),
                                                      transforms.Grayscale(),
                                                      transforms.ToTensor()])

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

    @staticmethod
    def euclidean_distance(x, y):
        """
        Calculates the Euclidean distance, between two values.
        :param x:
        :param y:
        :return: Euclidian distance.
        """

        return torch.norm(x - y)

    def get_query_vector(self):
        medicine_classes = os.listdir(CONST.dir_query_rgb)
        query_vectors = []
        labels = []

        for med_class in tqdm(medicine_classes, desc="Process query images"):
            query_image_paths_con = os.listdir(os.path.join(CONST.dir_query_contour, med_class))
            query_image_paths_rgb = os.listdir(os.path.join(CONST.dir_query_rgb, med_class))
            query_image_paths_tex = os.listdir(os.path.join(CONST.dir_query_texture, med_class))

            for idx, (con, rgb, tex) in enumerate(zip(query_image_paths_con, query_image_paths_rgb,
                                                      query_image_paths_tex)):

                con_query_image = Image.open(os.path.join(CONST.dir_query_contour, med_class, con))
                con_query_image = self.preprocess_con_tex(con_query_image)

                rgb_query_image = Image.open(os.path.join(CONST.dir_query_rgb, med_class, rgb))
                rgb_query_image = self.preprocess_rgb(rgb_query_image)

                tex_query_image = Image.open(os.path.join(CONST.dir_query_texture, med_class, tex))
                tex_query_image = self.preprocess_con_tex(tex_query_image)

                with torch.no_grad():
                    query_vector1 = self.network_con(con_query_image.unsqueeze(0)).squeeze()
                    query_vector2 = self.network_rgb(rgb_query_image.unsqueeze(0)).squeeze()
                    query_vector3 = self.network_tex(tex_query_image.unsqueeze(0)).squeeze()
                query_vector = torch.cat((query_vector1, query_vector2, query_vector3), dim=0)
                query_vectors.append(query_vector)
                labels.append(med_class)

        return query_vectors, labels

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- G E T   R E F   V E C ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
    def get_ref_vectors(self):
        medicine_classes = os.listdir(CONST.dir_bounding_box)
        reference_vectors = []
        label = []

        for med_class in tqdm(medicine_classes, desc="Process reference images"):
            reference_image_paths_con = os.listdir(os.path.join(CONST.dir_contour, med_class))
            reference_image_paths_rgb = os.listdir(os.path.join(CONST.dir_bounding_box, med_class))
            reference_image_paths_tex = os.listdir(os.path.join(CONST.dir_texture, med_class))

            for idx, (con, rgb, tex) in enumerate(zip(reference_image_paths_con, reference_image_paths_rgb,
                                                      reference_image_paths_tex)):
                con_ref_image = Image.open(os.path.join(CONST.dir_contour, med_class, con))
                con_ref_image = self.preprocess_con_tex(con_ref_image)

                rgb_ref_image = Image.open(os.path.join(CONST.dir_bounding_box, med_class, rgb))
                rgb_ref_image = self.preprocess_rgb(rgb_ref_image)

                tex_ref_image = Image.open(os.path.join(CONST.dir_texture, med_class, tex))
                tex_ref_image = self.preprocess_con_tex(tex_ref_image)

                with torch.no_grad():
                    reference_vector1 = self.network_con(con_ref_image.unsqueeze(0)).squeeze()
                    reference_vector2 = self.network_rgb(rgb_ref_image.unsqueeze(0)).squeeze()
                    reference_vector3 = self.network_tex(tex_ref_image.unsqueeze(0)).squeeze()
                reference_vector = torch.cat((reference_vector1, reference_vector2, reference_vector3), dim=0)
                reference_vectors.append(reference_vector)
                label.append(med_class)

        return reference_vectors, label

    @staticmethod
    def measure_similarity_cos_sim(q_labels, r_labels, reference_vectors, query_vectors):
        similarity_scores_cosine = []
        predicted_medicine = []
        corresp_sim = []

        for idx_query, query_vector in tqdm(enumerate(query_vectors), total=len(query_vectors), desc="Query vectors"):
            scores = []
            for idx_ref, reference_vector in enumerate(reference_vectors):
                score = torch.nn.functional.cosine_similarity(query_vector, reference_vector, dim=0).item()
                scores.append(score)

            similarity_scores_cosine.append(scores)

            most_similar_indices = [scores.index(max(scores)) for scores in similarity_scores_cosine]
            most_similar_indices_and_scores = [(i, max(scores)) for i, scores in
                                               enumerate(similarity_scores_cosine)]

            predicted_medicine.append(r_labels[most_similar_indices[idx_query]])
            corresp_sim.append(most_similar_indices_and_scores[idx_query][1])

        df = pd.DataFrame(list(zip(q_labels, predicted_medicine, corresp_sim)),
                          columns=['GT Medicine Name', 'Predicted Medicine Name', 'Cosine similarity'])

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print(df)

        return q_labels, predicted_medicine

    def measure_similarity_euc_dist(self, q_labels, r_labels, reference_vectors, query_vectors):
        similarity_scores_euclidean = []
        predicted_medicine = []
        corresp_sim = []

        for idx_query, query_vector in tqdm(enumerate(query_vectors), total=len(query_vectors), desc="Query vectors"):
            scores = []
            for idx_ref, reference_vector in enumerate(reference_vectors):
                score = self.euclidean_distance(query_vector, reference_vector)
                scores.append(score)

            similarity_scores_euclidean.append(scores)

            most_similar_indices = [scores.index(min(scores)) for scores in similarity_scores_euclidean]
            most_similar_indices_and_scores = [(i, min(scores)) for i, scores in
                                               enumerate(similarity_scores_euclidean)]

            predicted_medicine.append(r_labels[most_similar_indices[idx_query]])
            corresp_sim.append(most_similar_indices_and_scores[idx_query][1])

        df = pd.DataFrame(list(zip(q_labels, predicted_medicine, corresp_sim)),
                          columns=['GT Medicine Name', 'Predicted Medicine Name', 'Euclidean distance'])

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print(df)

        return q_labels, predicted_medicine


    @staticmethod
    def draw_results_on_image(key, value):
        image = os.path.join(CONST.dir_test_images, '054_algoflex_s1_5_a_f4.png')
        mask = os.path.join(CONST.dir_unet_output, '054_algoflex_s1_5_a_f4_OUT.png')
        segment_pills(image, mask, key, value)

    @staticmethod
    def measure_accuracy(gt, pred):
        count = 0
        for i in range(len(gt)):
            if gt[i] == pred[i]:
                count += 1
        print(f"Accuracy: {count / len(gt)}")

    def main(self):
        query_vecs, q_labels = self.get_query_vector()
        ref_vecs, r_labels = self.get_ref_vectors()

        gt, pred = self.measure_similarity_cos_sim(q_labels, r_labels, ref_vecs, query_vecs)
        self.measure_accuracy(gt, pred)

        gt, pred = self.measure_similarity_euc_dist(q_labels, r_labels, ref_vecs, query_vecs)
        self.measure_accuracy(gt, pred)

        # self.draw_results_on_image(key, value)


if __name__ == "__main__":
    try:
        pill_rec = PillRecognition()
        pill_rec.main()
    except KeyboardInterrupt as kie:
        print(kie)
