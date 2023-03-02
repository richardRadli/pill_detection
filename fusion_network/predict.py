import os
import pandas as pd
import re
import torch
import torch.nn.functional as func

from torchvision import transforms
from PIL import Image

from const import CONST
from config import ConfigStreamNetwork
from stream_network.stream_network import StreamNetwork
from utils.utils import BColors, find_latest_file, segment_pills

cfg = ConfigStreamNetwork().parse()


class PillRecognition:
    def __init__(self):
        self.preprocess_rgb = None
        self.preprocess_con_tex = None
        self.query_image_tex = None
        self.query_image_rgb = None
        self.query_image_con = None
        self.network_con, self.network_rgb, self.network_tex = self.load_networks()

    @staticmethod
    def load_networks():
        """
        This function loads the pretrained networks, with the latest .pt files
        :return: The contour, rgb, and texture networks.
        """

        list_of_channels_tex_con = [1, 32, 48, 64, 128, 192, 256]
        list_of_channels_rgb = [3, 64, 96, 128, 256, 384, 512]

        network_con_path = CONST.dir_stream_contour_model_weights
        latest_con_pt_file = find_latest_file(network_con_path)

        network_rgb_path = CONST.dir_stream_rgb_model_weights
        latest_rgb_pt_file = find_latest_file(network_rgb_path)

        network_tex_path = CONST.dir_stream_texture_model_weights
        latest_tex_pt_file = find_latest_file(network_tex_path)

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

    def get_query_image(self):
        """
        Collects the query image, makes some transformation.
        :return:
        """

        query_image_path_con = os.path.join(CONST.dir_contour,
                                            'advilultraforte/015_advilultraforte_s1_5_a_f4_j.png').replace("\\", "/")
        query_image_path_rgb = os.path.join(CONST.dir_bounding_box,
                                            'advilultraforte/bbox_015_advilultraforte_s1_5_a_f4_j.png').replace("\\",
                                                                                                                "/")
        query_image_path_tex = os.path.join(CONST.dir_texture,
                                            'advilultraforte/015_advilultraforte_s1_5_a_f4_j.png').replace("\\", "/")

        self.preprocess_rgb = transforms.Compose([transforms.Resize((cfg.img_size, cfg.img_size)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.preprocess_con_tex = transforms.Compose([transforms.Resize((cfg.img_size, cfg.img_size)),
                                                      transforms.Grayscale(),
                                                      transforms.ToTensor()])

        query_image_con = Image.open(query_image_path_con)
        self.query_image_con = self.preprocess_con_tex(query_image_con)

        query_image_rgb = Image.open(query_image_path_rgb)
        self.query_image_rgb = self.preprocess_rgb(query_image_rgb)

        query_image_tex = Image.open(query_image_path_tex)
        self.query_image_tex = self.preprocess_con_tex(query_image_tex)

    def get_query_vector(self):
        """
        Calculates 
        :return:
        """

        with torch.no_grad():
            query_vector1 = self.network_con(self.query_image_con.unsqueeze(0)).squeeze()
            query_vector2 = self.network_rgb(self.query_image_rgb.unsqueeze(0)).squeeze()
            query_vector3 = self.network_tex(self.query_image_tex.unsqueeze(0)).squeeze()
        return torch.cat((query_vector1, query_vector2, query_vector3), dim=0)

    def get_ref_vectors(self):
        reference_image_paths_con = os.listdir("E:/users/ricsi/IVM/images/ref_images/cont")
        reference_image_paths_rgb = os.listdir("E:/users/ricsi/IVM/images/ref_images/rgb")
        reference_image_paths_tex = os.listdir("E:/users/ricsi/IVM/images/ref_images/tex")

        reference_vectors = []
        labels = []

        for idx, (con, rgb, tex) in enumerate(zip(reference_image_paths_con, reference_image_paths_rgb,
                                                  reference_image_paths_tex)):
            match = re.search(r"\d{3,4}_(.+)_s", con)
            labels.append(match.group(1))

            con_ref_image = Image.open(os.path.join("E:/users/ricsi/IVM/images/ref_images/cont", con))
            con_ref_image = self.preprocess_con_tex(con_ref_image)

            rgb_ref_image = Image.open(os.path.join("E:/users/ricsi/IVM/images/ref_images/rgb", rgb))
            rgb_ref_image = self.preprocess_rgb(rgb_ref_image)

            tex_ref_image = Image.open(os.path.join("E:/users/ricsi/IVM/images/ref_images/tex", tex))
            tex_ref_image = self.preprocess_con_tex(tex_ref_image)

            with torch.no_grad():
                reference_vector1 = self.network_con(con_ref_image.unsqueeze(0)).squeeze()
                reference_vector2 = self.network_rgb(rgb_ref_image.unsqueeze(0)).squeeze()
                reference_vector3 = self.network_tex(tex_ref_image.unsqueeze(0)).squeeze()
            reference_vector = torch.cat((reference_vector1, reference_vector2, reference_vector3), dim=0)
            reference_vectors.append(reference_vector)

        return reference_vectors, labels

    def measure_similarity(self, labels, reference_vectors, query_vector):
        similarity_scores_cosine = []
        for reference_vector in reference_vectors:
            similarity_score = func.cosine_similarity(query_vector.unsqueeze(0), reference_vector.unsqueeze(0)).item()
            similarity_scores_cosine.append(similarity_score)

        similarity_scores_euclidean = []
        for reference_vector in reference_vectors:
            similarity_score = self.euclidean_distance(query_vector, reference_vector).item()
            similarity_scores_euclidean.append(similarity_score)

        df = pd.DataFrame(list(zip(labels, similarity_scores_cosine, similarity_scores_euclidean)),
                          columns=['Medicine Name', 'Cosine similarity', 'Euclidian distance'])

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print(df)

        return labels, similarity_scores_euclidean, similarity_scores_cosine

    @staticmethod
    def find_best_match(labels, similarity_scores, sim_type):
        my_dict = dict(zip(labels, similarity_scores))

        # Find the maximum value and its corresponding key
        if sim_type == "euclidian":
            key = min(my_dict, key=my_dict.get)
            value = my_dict[key]
            print(f"The best match is {BColors.HEADER}{key}{BColors.ENDC} with {value:.4f} Euclidian distance")
        elif sim_type == "cosine":
            key = max(my_dict, key=my_dict.get)
            value = my_dict[key]
            print(f"The best match is {BColors.HEADER}{key}{BColors.ENDC} with {value:.4%} cosine similarity score")
        else:
            raise ValueError("Wrong type!")

        return key, value

    @staticmethod
    def draw_results_on_image(key, value):
        image = os.path.join(CONST.dir_test_images, '015_advilultraforte_s1_5_a_f4_j.png')
        mask = os.path.join(CONST.dir_unet_output, ('015_advilultraforte_s1_5_a_f4_j' + '_OUT' + '.png'))
        segment_pills(image, mask, key, value)

    def main(self):
        self.get_query_image()
        query_vector = self.get_query_vector()
        ref_vecs, labels = self.get_ref_vectors()
        labels, sim_euc, sim_cos = self.measure_similarity(labels, ref_vecs, query_vector)
        print("\n")
        self.find_best_match(labels, sim_euc, "euclidian")
        key, value = self.find_best_match(labels, sim_cos, "cosine")
        self.draw_results_on_image(key, value)


if __name__ == "__main__":
    pill_rec = PillRecognition()
    pill_rec.main()
