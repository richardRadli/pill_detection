import os
import torch

from typing import Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++ S T R E A M   D A T A S E T ++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class StreamDataset(Dataset):
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- _ I N I T _ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, dataset_dir: str, type_of_network: str) -> None:
        """
        This function is the constructor of a class. The class takes two arguments, dataset_dir and type_of_network,
        which specify the directory of the dataset and the type of network to use (either RGB, Contour, or Texture).

        The function initializes some instance variables and sets the appropriate transformation pipeline depending on
        the type of network specified. It also maps the classes in the dataset to their respective index, and stores
        the file paths and class indices of all samples in the dataset as a list of tuples.

        :param dataset_dir:
        :param type_of_network:
        """

        self.dataset_dir = dataset_dir
        self.type_of_network = type_of_network

        if self.type_of_network == "RGB":
            self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        elif self.type_of_network in ["Contour", "Texture"]:
            self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError("Wrong kind of network")

        self.classes = os.listdir(self.dataset_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.dataset_dir, class_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                self.samples.append((file_path, class_idx))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ _ G E T I T E M _ -----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index: int) -> Tuple:
        """
        This function returns a tuple of three images: anchor, positive, and negative. It takes an index as an argument,
        which is used to obtain the anchor image and its corresponding class from the "self.samples" list.
        Then, the function uses the _get_same_class_sample and _get_different_class_sample methods to get the path of
        positive and negative images, respectively. The _load_image method is called to load the images from the given
        paths, and the three images are returned as a tuple.
        The anchor and positive images belong to the same class, while the negative image belongs to a different
        class.

        :param index:
        :return:
        """

        anchor_path, anchor_class = self.samples[index]
        positive_path, _ = self._get_same_class_sample(anchor_class)
        negative_path, _ = self._get_different_class_sample(anchor_class)
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)

        return anchor, positive, negative

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- _ L E N _ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        """
        :return: the number of samples in the dataset.
        """

        return len(self.samples)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ G E T   S A M E   C L A S S   S A M P L E -----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _get_same_class_sample(self, anchor_class_idx: int) -> Tuple:
        """
        This private method is used by the TripletImageDataset class to obtain a sample of an image that belongs to the
        same class as the anchor image. The method takes an integer anchor_class_idx that represents the class index
        of the anchor image.

        :param anchor_class_idx:
        :return:
        """

        while True:
            idx = torch.randint(len(self), (1,)).item()
            path, class_idx = self.samples[idx]
            if class_idx == anchor_class_idx:
                return path, class_idx

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ G E T   D I F F   C L A S S   S A M P L E -----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _get_different_class_sample(self, anchor_class_idx: int) -> Tuple:
        """
        This function is a helper function used by the TripletDataset class to obtain a random sample that belongs
        to a different class than the anchor sample.

        :param anchor_class_idx:
        :return:
        """

        while True:
            idx = torch.randint(len(self), (1,)).item()
            path, class_idx = self.samples[idx]
            if class_idx != anchor_class_idx:
                return path, class_idx

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- L O A D   I M A G E ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _load_image(self, path: str) -> Image:
        """
        This function loads an image from a given path and returns it after applying some image transformations
        specified by the transform attribute in the class constructor. The specific image transformations depend on
        the type of network specified in the constructor.

        :param path:
        :return:
        """

        with open(path, 'rb') as f:
            if self.type_of_network == "RGB":
                img = Image.open(f)
            elif self.type_of_network in ["Contour", "Texture"]:
                img = Image.open(f).convert("L")
            else:
                raise ValueError("Wrong type of network!")

            return self.transform(img)
