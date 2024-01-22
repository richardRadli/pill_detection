import logging
import os
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from PIL import Image

from config.config import ConfigTestingUnet
from dataloader_unet import BasicDataset

cfg = ConfigTestingUnet().parse()


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if cfg.classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = "C:/Users/ricsi/Documents/project/storage/IVM/datasets/nih/ref/00002322730/BKMNRTQ6_7JVE3M5_DIU81YSSX!P!HB.JPG"
    out_filename = "C:/Users/ricsi/Desktop/unet.jpg"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=cfg.channels,
        classes=cfg.classes
    )

    # Modify the last layer for fine-tuning
    num_channels = net.segmentation_head[0].in_channels
    net.segmentation_head[0] = \
        torch.nn.Conv2d(in_channels=num_channels, out_channels=cfg.classes, kernel_size=1).to(device)

    logging.info(f'Loading model')
    logging.info(f'Using device {device}')

    model = "C:/Users/ricsi/Documents/project/storage/IVM/data/unet/weights/epoch_5.pt"

    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    img = Image.open(in_files)

    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=cfg.scale,
                       out_threshold=cfg.mask_threshold,
                       device=device)

    result = mask_to_image(mask, mask_values)
    result.save(out_filename)
    logging.info(f'Mask saved to {out_filename}')