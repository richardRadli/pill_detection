import os

from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataLoader(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        self.images_dir = images_dir
        self.mask_dir = masks_dir
        self.transform = transform
        self.images_names = os.listdir(images_dir)

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        image_name = self.images_names[idx]
        image_path = os.path.join(self.images_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image_size = image.size

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, image_size
