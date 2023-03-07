import logging
import numpy as np
import os
import torch
import torch.nn.functional as F

from PIL import Image

from config import ConfigTestingUnet
from unet import UNet
from unet.data_loading import BasicDataset
from utils.utils import plot_img_and_mask

cfg = ConfigTestingUnet().parse()


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- P R E D I C T   I M G ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------- G E T   O U T P U T   F I L E N A M E S --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def get_output_filenames():
    """
    This function returns a list of output file names that will be produced after running some image processing on the
    input files specified in the cfg configuration object.

    :return: a list of output file names.
    """

    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return cfg.output or list(map(_generate_name, cfg.input))


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- M A S K   T O   I M A G E ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ M A I N -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = cfg.input
    out_files = get_output_filenames()

    net = UNet(cfg.channels, cfg.classes, bilinear=cfg.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {cfg.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(cfg.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=cfg.scale,
                           out_threshold=cfg.mask_threshold,
                           device=device)

        if not cfg.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if cfg.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)


if __name__ == '__main__':
    main()
