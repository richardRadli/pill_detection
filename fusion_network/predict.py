import os
import re
import torch
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

from const import CONST
from stream_network.stream_network import StreamNetwork
from utils.utils import find_latest_file


class PillRecognition:
    def __init__(self):
        self.query_image_tex = None
        self.query_image_rgb = None
        self.query_image_con = None
        self.network_con, self.network_rgb, self.network_tex = self.load_networks()

    @staticmethod
    def load_networks():
        list_of_channels_tex_con = [1, 32, 48, 64, 128, 192, 256]
        list_of_channels_rgb = [3, 64, 96, 128, 256, 384, 512]

        # Define the paths to the `.pt` files that contain the state dicts of the trained networks.
        network_con_path = CONST.dir_stream_contour_model_weights
        latest_con_pt_file = find_latest_file(network_con_path)

        network_rgb_path = CONST.dir_stream_rgb_model_weights
        latest_rgb_pt_file = find_latest_file(network_rgb_path)

        network_tex_path = CONST.dir_stream_texture_model_weights
        latest_tex_pt_file = find_latest_file(network_tex_path)

        network_con = StreamNetwork(loc=list_of_channels_tex_con)
        network_rgb = StreamNetwork(loc=list_of_channels_rgb)
        network_tex = StreamNetwork(loc=list_of_channels_tex_con)

        # Load the state dicts of the trained networks from the `.pt` files.
        network_con.load_state_dict(torch.load(latest_con_pt_file))
        network_rgb.load_state_dict(torch.load(latest_rgb_pt_file))
        network_tex.load_state_dict(torch.load(latest_tex_pt_file))

        return network_con, network_rgb, network_tex

    @staticmethod
    def euclidean_distance(x, y):
        return torch.norm(x - y)

    def get_query_image(self):
        query_image_path_con = os.path.join(CONST.dir_contour,
                                            'advilultraforte/003_advilultraforte_s1_2_a_f4_b.png').replace("\\", "/")
        query_image_path_rgb = os.path.join(CONST.dir_bounding_box,
                                            'advilultraforte/bbox_003_advilultraforte_s1_2_a_f4_b.png').replace("\\",
                                                                                                                "/")
        query_image_path_tex = os.path.join(CONST.dir_texture,
                                            'advilultraforte/003_advilultraforte_s1_2_a_f4_b.png').replace("\\", "/")

        self.preprocess_rgb = transforms.Compose([transforms.Resize((128, 128)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.preprocess_con_tex = transforms.Compose([transforms.Resize((128, 128)),
                                                 transforms.Grayscale(),
                                                 transforms.ToTensor()])

        query_image_con = Image.open(query_image_path_con)
        self.query_image_con = self.preprocess_con_tex(query_image_con)

        query_image_rgb = Image.open(query_image_path_rgb)
        self.query_image_rgb =self. preprocess_rgb(query_image_rgb)

        query_image_tex = Image.open(query_image_path_tex)
        self.query_image_tex = self.preprocess_con_tex(query_image_tex)

    def get_query_vector(self):
        with torch.no_grad():
            query_vector1 = self.network_con(self.query_image_con.unsqueeze(0)).squeeze()
            query_vector2 = self.network_rgb(self.query_image_rgb.unsqueeze(0)).squeeze()
            query_vector3 = self.network_tex(self.query_image_tex.unsqueeze(0)).squeeze()
        return torch.cat((query_vector1, query_vector2, query_vector3), dim=0)

    def get_ref_vectors(self):
        reference_image_paths_con = os.listdir(CONST.dir_contour + "/advilultraforte")
        reference_image_paths_rgb = os.listdir(CONST.dir_bounding_box + "/advilultraforte")
        reference_image_paths_tex = os.listdir(CONST.dir_texture + "/advilultraforte")

        reference_vectors = []
        labels = []

        for idx, (con, rgb, tex) in enumerate(zip(reference_image_paths_con, reference_image_paths_rgb,
                                                  reference_image_paths_tex)):
            match = re.search(r"\d{3,4}_(.+)_s", con)
            labels.append(match.group(1))

            con_ref_image = Image.open(os.path.join(CONST.dir_contour, "advilultraforte", con))
            con_ref_image = self.preprocess_con_tex(con_ref_image)

            rgb_ref_image = Image.open(os.path.join(CONST.dir_bounding_box, "advilultraforte", rgb))
            rgb_ref_image = self.preprocess_rgb(rgb_ref_image)

            tex_ref_image = Image.open(os.path.join(CONST.dir_texture, "advilultraforte", tex))
            tex_ref_image = self.preprocess_con_tex(tex_ref_image)

            # Pass the reference image through each of the three networks and concatenate their output vectors.
            with torch.no_grad():
                reference_vector1 = self.network_con(con_ref_image.unsqueeze(0)).squeeze()
                reference_vector2 = self.network_rgb(rgb_ref_image.unsqueeze(0)).squeeze()
                reference_vector3 = self.network_tex(tex_ref_image.unsqueeze(0)).squeeze()
            reference_vector = torch.cat((reference_vector1, reference_vector2, reference_vector3), dim=0)
            reference_vectors.append(reference_vector)

        return reference_vectors, labels

    def measure_similarity(self, labels, reference_vectors, query_vector):
        # Measure the cosine similarity between the query vector and each of the reference vectors.
        similarity_scores = []
        for reference_vector in reference_vectors:
            similarity_score = F.cosine_similarity(query_vector.unsqueeze(0), reference_vector.unsqueeze(0)).item()
            similarity_scores.append(similarity_score)

        print("\nCosine similarity")
        for idx, (l, s) in enumerate(zip(labels, similarity_scores)):
            print(f"{l}: {s:.4%}")

        similarity_scores = []
        for reference_vector in reference_vectors:
            similarity_score = self.euclidean_distance(query_vector, reference_vector).item()
            similarity_scores.append(similarity_score)

        print("\nEuclidean distance")
        # Print the similarity scores.
        for idx, (l, s) in enumerate(zip(labels, similarity_scores)):
            print(f"{l}: {s:.4f}")

    def main(self):
        self.get_query_image()
        query_vector = self.get_query_vector()
        ref_vecs, labels = self.get_ref_vectors()
        self.measure_similarity(labels, ref_vecs, query_vector)


if __name__ == "__main__":
    pill_rec = PillRecognition()
    pill_rec.main()

#
