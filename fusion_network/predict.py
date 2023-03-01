import os
import torch

from torchvision import transforms
from PIL import Image

from stream_network.stream_network import StreamNetwork

list_of_channels_tex_con = [1, 32, 48, 64, 128, 192, 256]
list_of_channels_rgb = [3, 64, 96, 128, 256, 384, 512]

# Define the paths to the `.pt` files that contain the state dicts of the trained networks.
network1_path = 'C:/Users/rrb12/Downloads/pt/con_epoch_195.pt'
network2_path = 'C:/Users/rrb12/Downloads/pt/rgb_epoch_190.pt'
network3_path = 'C:/Users/rrb12/Downloads/pt/tex_epoch_195.pt'

network_con = StreamNetwork(loc=list_of_channels_tex_con)
network_rgb = StreamNetwork(loc=list_of_channels_rgb)
network_tex = StreamNetwork(loc=list_of_channels_tex_con)

# Load the state dicts of the trained networks from the `.pt` files.
network_con.load_state_dict(torch.load(network1_path))
network_rgb.load_state_dict(torch.load(network2_path))
network_tex.load_state_dict(torch.load(network3_path))


# Assume you have a set of reference images and three separate networks
ref_images_contour = os.listdir("C:/Users/rrb12/Downloads/imgs/cont/")
ref_images_rgb = os.listdir("C:/Users/rrb12/Downloads/imgs/rgb/")
ref_images_texture = os.listdir("C:/Users/rrb12/Downloads/imgs/tex/")

# # Load the query image and apply any necessary preprocessing (e.g., resizing, normalization).
query_image_path_con = 'C:/Users/rrb12/Downloads/imgs/cont/003_advilultraforte_s1_2_a_f4_b.png'
query_image_path_rgb = 'C:/Users/rrb12/Downloads/imgs/rgb/003_advilultraforte_s1_2_a_f4_b.png'
query_image_path_tex = 'C:/Users/rrb12/Downloads/imgs/tex/003_advilultraforte_s1_2_a_f4_b.png'

preprocess_rgb = transforms.Compose([transforms.Resize((128, 128)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

preprocess_con_tex = transforms.Compose([transforms.Resize((128, 128)),
                                         transforms.Grayscale(),
                                         transforms.ToTensor()])

query_image_con = Image.open(query_image_path_con)
query_image_con = preprocess_con_tex(query_image_con)

query_image_rgb = Image.open(query_image_path_rgb)
query_image_rgb = preprocess_rgb(query_image_rgb)

query_image_tex = Image.open(query_image_path_tex)
query_image_tex = preprocess_con_tex(query_image_tex)

# Pass the query image through each of the three networks and concatenate their output vectors.
with torch.no_grad():
    query_vector1 = network_con(query_image_con.unsqueeze(0)).squeeze()
    query_vector2 = network_rgb(query_image_rgb.unsqueeze(0)).squeeze()
    query_vector3 = network_tex(query_image_tex.unsqueeze(0)).squeeze()
query_vector = torch.cat((query_vector1, query_vector2, query_vector3), dim=0)

# Load the reference images and apply any necessary preprocessing (e.g., resizing, normalization).
reference_image_paths_con = os.listdir("C:/Users/rrb12/Downloads/imgs/cont/")
reference_image_paths_rgb = os.listdir("C:/Users/rrb12/Downloads/imgs/rgb/")
reference_image_paths_tex = os.listdir("C:/Users/rrb12/Downloads/imgs/tex/")

reference_vectors = []
for idx, (con, rgb, tex) in enumerate(zip(reference_image_paths_con, reference_image_paths_rgb, reference_image_paths_tex)):
    con_ref_image = Image.open(os.path.join("C:/Users/rrb12/Downloads/imgs/cont/", con))
    con_ref_image = preprocess_con_tex(con_ref_image)

    rgb_ref_image = Image.open(os.path.join("C:/Users/rrb12/Downloads/imgs/rgb/", rgb))
    rgb_ref_image = preprocess_rgb(rgb_ref_image)

    tex_ref_image = Image.open(os.path.join("C:/Users/rrb12/Downloads/imgs/tex/", tex))
    tex_ref_image = preprocess_con_tex(tex_ref_image)

    # Pass the reference image through each of the three networks and concatenate their output vectors.
    with torch.no_grad():
        reference_vector1 = network_con(con_ref_image.unsqueeze(0)).squeeze()
        reference_vector2 = network_rgb(rgb_ref_image.unsqueeze(0)).squeeze()
        reference_vector3 = network_tex(tex_ref_image.unsqueeze(0)).squeeze()
    reference_vector = torch.cat((reference_vector1, reference_vector2, reference_vector3), dim=0)
    reference_vectors.append(reference_vector)

# Measure the cosine similarity between the query vector and each of the reference vectors.
similarity_scores = []
for reference_vector in reference_vectors:
    similarity_score = torch.nn.functional.cosine_similarity(query_vector.unsqueeze(0), reference_vector.unsqueeze(0)).item()
    similarity_scores.append(similarity_score)

print(similarity_scores)

