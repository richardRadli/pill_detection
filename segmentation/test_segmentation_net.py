import colorama
import gc
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchinfo import summary
from tqdm import tqdm
from sklearn.metrics import jaccard_score

from config.json_config import json_config_selector
from config.dataset_paths_selector import dataset_images_path_selector
from config.networks_paths_selector import segmentation_paths
from data_loader_segmentation_net import SegmentationDataLoader
from segmentation_network_models.segmentation_network_selector import SegmentationNetworkFactory
from utils.utils import (create_timestamp, load_config_json, find_latest_file_in_latest_directory, use_gpu_if_available,
                         setup_logger)


class TestUnet:
    def __init__(self):
        timestamp = create_timestamp()
        colorama.init()
        setup_logger()

        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("segmentation_net").get("schema"),
                json_filename=json_config_selector("segmentation_net").get("config")
            )
        )

        network_type = self.cfg.get("network_type")
        dataset_name = self.cfg.get("dataset_name")
        self.threshold = self.cfg.get("mask_threshold")

        self.weight_files_dir = segmentation_paths(network_type, dataset_name).get("weights_folder")

        self.compare_dir = (
            self.create_folder(
                network_type, 
                dataset_name, 
                "prediction_folder", 
                "compare", 
                timestamp
            )
        )

        self.out_dir = (
            self.create_folder(
                network_type, 
                dataset_name, 
                "prediction_folder", 
                "out", 
                timestamp
            )
        )

        if self.cfg.get("seed"):
            seed_number = 1234
            torch.manual_seed(seed_number)
            torch.cuda.manual_seed(seed_number)

        self.device = (
            use_gpu_if_available()
        )

        self.model = (
            self.load_model(network_type)
        )

        self.test_dataset = (
            self.create_segmentation_dataset(
                images_dir=dataset_images_path_selector(dataset_name).get("test").get("test_images"),
                masks_dir=dataset_images_path_selector(dataset_name).get("test").get("test_masks"),
                batch_size=self.cfg.get("batch_size"),
                shuffle=False
            )
        )
    
    @staticmethod
    def create_folder(network_type, dataset_name, dir1, dir2, timestamp):
        """

        Args:
            network_type: 
            dataset_name: 
            dir1: 
            dir2:
            timestamp

        Returns:

        """
        
        directory = segmentation_paths(network_type, dataset_name).get(dir1).get(dir2)
        directory = os.path.join(directory, timestamp)
        os.makedirs(directory, exist_ok=True)
        return directory
        
    def load_model(self, network_type):
        """

        Return:
        """

        model = (
            SegmentationNetworkFactory().create_model(
                network_type=network_type,
                cfg=self.cfg
            )
        ).to(self.device)
        summary(model)

        latest_file = find_latest_file_in_latest_directory(path=self.weight_files_dir)
        model.load_state_dict(
            torch.load(
                latest_file
            )
        )

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

        dataset = (
            SegmentationDataLoader(
                images_dir=images_dir,
                masks_dir=masks_dir,
                transform=transform
            )
        )

        dataloader = (
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle
            )
        )

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

        with (torch.no_grad()):
            for i, (images, masks, image_size) in tqdm(enumerate(self.test_dataset),
                                                       total=len(self.test_dataset),
                                                       desc="Evaluation"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                prediction = self.model(images)
                prediction = torch.sigmoid(prediction)
                prediction = (prediction > self.threshold).float()
                masks = (masks > self.threshold).float()

                for j in range(images.size(0)):
                    original_size = (image_size[0][j].item(), image_size[1][j].item())
                    pred_mask = F.interpolate(
                        prediction[j].unsqueeze(0), size=original_size, mode='nearest'
                    ).squeeze().cpu().numpy()

                    true_mask = F.interpolate(
                        masks[j].unsqueeze(0), size=original_size, mode='nearest'
                    ).squeeze().cpu().numpy()

                    pred_mask_image = to_pil(torch.from_numpy(pred_mask).float())

                    self.save_prediction(
                        save_filename=f"{self.out_dir}/{i}_{j}.jpg",
                        predicted_image=pred_mask_image
                    )

                    self.save_compare(
                        filename=f"{self.compare_dir}/{i}_{j}.jpg",
                        true_mask=true_mask,
                        pred_mask=pred_mask
                    )

                    jc = (
                        self.calculate_jaccard_score(
                            jaccard_scores, i, j, images, true_mask, pred_mask
                        )
                    )
                    jaccard_scores.append(jc)

        # Calculate and log the average Jaccard score
        avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
        logging.info(f"Average Jaccard Score: {avg_jaccard:.4f}")


if __name__ == "__main__":
    test_unet = TestUnet()
    test_unet.make_prediction()
