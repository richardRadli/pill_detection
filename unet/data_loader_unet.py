import logging
import numpy as np
import torch

from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import setup_logger


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        setup_logger()

        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file))
                    and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(self.unique_mask_values,
                               mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids), total=len(self.ids)))

        self.mask_values = np.unique(np.concatenate(unique), axis=0).tolist()

        logging.info(f'Unique mask values: {self.mask_values}')

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- __L E N__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self):
        return len(self.ids)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ L O A D   I M A G E ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def load_image(filename: str) -> Image:
        """
        Load an image from a file with support for different file formats

        :param filename: path to the file to load
        :return: the loaded image
        """

        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ U N I Q U E   M A S K   V A L U E S -----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def unique_mask_values(self, idx: str, mask_dir: Path, mask_suffix: str) -> np.ndarray:
        """
        Get unique values in a mask file for a given ID.

        :param idx: ID of the image/mask
        :param mask_dir: Directory where mask files are located
        :param mask_suffix: Suffix of the mask file
        :return: Numpy array of unique values in the mask
        """

        mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
        mask = np.asarray(self.load_image(str(mask_file)))
        if mask.ndim == 2:
            return np.unique(mask)
        elif mask.ndim == 3:
            mask = mask.reshape(-1, mask.shape[-1])
            return np.unique(mask, axis=0)
        else:
            raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- P R E P R O C E S S ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        new_w, new_h = int(scale * w), int(scale * h)

        # Adjust new width and height to be divisible by 32
        if new_w % 32 != 0:
            new_w = (new_w // 32 + 1) * 32
        if new_h % 32 != 0:
            new_h = (new_h // 32 + 1) * 32

        assert new_w > 0 and new_h > 0, 'Scale is too small, resized images would have no pixel'

        # Perform resizing of the input using pil_img.resize
        pil_img = pil_img.resize((new_w, new_h), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            # Handle mask processing
            mask = np.zeros((new_h, new_w), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask
        else:
            # Handle image processing
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(str(name) + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(str(name) + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load_image(str(mask_file[0]))
        img = self.load_image(str(img_file[0]))

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class UNetDataLoader(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')