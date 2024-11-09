import colorama
import gc
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import segmentation_models_pytorch as smp
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage, InterpolationMode
from torchvision.transforms import functional as tf
from tqdm import tqdm
from sklearn.metrics import jaccard_score

from config.json_config import json_config_selector
from config.dataset_paths_selector import dataset_images_path_selector
from config.networks_paths_selector import segmentation_paths
from data_loader_segmentation_net import SegmentationDataLoader
from utils.utils import (create_timestamp, load_config_json, find_latest_file_in_latest_directory, use_gpu_if_available,
                         setup_logger)


class TestUnet:
    def __init__(self):
        timestamp = create_timestamp()
        colorama.init()
        setup_logger()

        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("unet").get("schema"),
                json_filename=json_config_selector("unet").get("config")
            )
        )
        dataset_name = self.cfg.get("dataset_name")

        self.weight_files_dir = segmentation_paths(dataset_name).get("weights_folder")

        compare_dir = segmentation_paths(dataset_name).get("prediction_folder").get("compare")
        self.unet_compare_dir = os.path.join(compare_dir, timestamp)
        os.makedirs(self.unet_compare_dir, exist_ok=True)

        out_dir = segmentation_paths(dataset_name).get("prediction_folder").get("out")
        self.unet_out_dir = os.path.join(out_dir, timestamp)
        os.makedirs(self.unet_out_dir, exist_ok=True)

        if self.cfg.get("seed"):
            torch.manual_seed(1234)

        self.device = (
            use_gpu_if_available()
        )

        self.model = (
            self.load_model()
        )

        self.test_dataset = (
            self.create_segmentation_dataset(
                images_dir=dataset_images_path_selector(dataset_name).get("test").get("images"),
                masks_dir=dataset_images_path_selector(dataset_name).get("test").get("mask_images"),
                batch_size=self.cfg.get("batch_size"),
                shuffle=False
            )
        )

    def load_model(self):
        """

        Return:
        """

        model = smp.Unet(
            encoder_name=self.cfg.get("encoder_name"),
            encoder_weights=self.cfg.get("encoder_weights"),
            in_channels=self.cfg.get("channels"),
            classes=self.cfg.get("classes")
        )

        latest_file = find_latest_file_in_latest_directory(path=self.weight_files_dir)
        model.load_state_dict(
            torch.load(
                latest_file
            )
        )

        model = model.to(self.device)
        model.eval()

        return model

    def create_segmentation_dataset(self, images_dir, masks_dir, batch_size, shuffle):
        """

            images_dir:
            masks_dir:
            batch_size:
            shuffle:

        Return:
        """

        transform = transforms.Compose([
            transforms.Resize((self.cfg.get("resized_img_size"), self.cfg.get("resized_img_size"))),
            transforms.ToTensor(),
        ])

        dataset = SegmentationDataLoader(images_dir=images_dir,
                                         masks_dir=masks_dir,
                                         transform=transform)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    @staticmethod
    def save_compare(filename, true_mask, pred_mask):
        """

        Args:
            filename:
            true_mask:
            pred_mask:

        Return:

        """

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(true_mask, cmap='gray')
        axes[0].set_title("Ground Truth Mask")
        axes[1].imshow(pred_mask, cmap='gray')
        axes[1].set_title("Predicted Mask")
        plt.savefig(filename)
        plt.close()
        gc.collect()

    @staticmethod
    def save_prediction(save_filename, predicted_image):
        """

            save_filename:
            predicted_image:

        Return:
        """

        predicted_image.save(save_filename)

    @staticmethod
    def calculate_jaccard_score(jaccard_scores, i, j, images, true_mask, pred_mask):
        """

            jaccard_scores:
            i:
            j:
            images:
            true_mask:
            pred_mask:

        Return:
        """

        pred_mask_flat = pred_mask.flatten()
        true_mask_flat = true_mask.flatten()

        jaccard = jaccard_score(true_mask_flat, pred_mask_flat, average='binary')
        jaccard_scores.append(jaccard)

        logging.info(f"Image {i * images.size(0) + j + 1}: Jaccard Score = {jaccard:.4f}")
        return jaccard_scores

    def make_prediction(self):
        jaccard_scores = []
        to_pil = ToPILImage()
        original_size = [1683, 2465]

        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(self.test_dataset)):
                images = images.to(self.device)
                masks = masks.to(self.device)
                prediction = self.model(images)
                prediction = torch.sigmoid(prediction)
                prediction = (prediction > 0.5).float()
                masks = (masks > 0.5).float()

                for j in range(images.size(0)):  # Iterate over each image in the batch
                    pred_mask = prediction[j].squeeze().cpu().numpy()
                    true_mask = masks[j].squeeze().cpu().numpy()

                    pred_mask_tensor = torch.from_numpy(pred_mask).unsqueeze(0)
                    pred_mask_resized = tf.resize(
                        pred_mask_tensor,
                        original_size,
                        interpolation=InterpolationMode.NEAREST
                    )

                    pred_mask_resized_image = to_pil(pred_mask_resized.squeeze())

                    self.save_prediction(
                        save_filename=f"{self.unet_out_dir}/{i}_{j}.jpg",
                        predicted_image=pred_mask_resized_image
                    )

                    self.save_compare(
                        filename=f"{self.unet_compare_dir}/{i}_{j}.jpg",
                        true_mask=true_mask,
                        pred_mask=pred_mask
                    )

                    jaccard_scores = (
                        self.calculate_jaccard_score(
                            jaccard_scores, i, j, images, true_mask, pred_mask
                        )
                    )

        avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
        logging.info(f"Average Jaccard Score: {avg_jaccard:.4f}")


if __name__ == "__main__":
    test_unet = TestUnet()
    test_unet.make_prediction()
