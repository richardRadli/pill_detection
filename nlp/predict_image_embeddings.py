import colorama
import os
import torch

from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from stream_network_models.efficient_net_b0 import EfficientNet


class Vectorization:
    def __init__(self):
        self.device = "cuda"
        self.preprocess_rgb = \
            transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def get_vectors(self, rgb_dir: str, network_rgb):
        color = colorama.Fore.BLUE
        medicine_classes = os.listdir(rgb_dir)
        vectors = []
        labels = []
        label_to_vectors = {}

        for image_name in tqdm(medicine_classes, desc=color + "\nProcess images"):
            image_paths_rgb = os.listdir(os.path.join(rgb_dir, image_name))
            directory_vectors = []

            for _, rgb in enumerate(image_paths_rgb):
                rgb_image = Image.open(os.path.join(rgb_dir, image_name, rgb))
                rgb_image = self.preprocess_rgb(rgb_image)

                # Make prediction
                with torch.no_grad():
                    rgb_image = rgb_image.unsqueeze(0).to(self.device)
                    rgb_vector = network_rgb(rgb_image).squeeze().cpu()
                    directory_vectors.append(rgb_vector)

            # Calculate mean vector for the directory
            if directory_vectors:
                mean_vector = torch.mean(torch.stack(directory_vectors), dim=0)
                vectors.append(mean_vector.numpy())
                labels.append(image_name)
                label_to_vectors[image_name] = mean_vector.numpy()

        torch.save(label_to_vectors, "C:/Users/ricsi/Desktop/image_embeddings.pt")
        return label_to_vectors

    def main(self):
        pt_file = "epoch_6.pt"
        model = EfficientNet(num_out_feature=256)
        model.to(self.device)
        model.load_state_dict(torch.load(pt_file))
        label_to_vectors = self.get_vectors("rgb", model)

        print(label_to_vectors)


if __name__ == '__main__':
    Vectorization().main()
