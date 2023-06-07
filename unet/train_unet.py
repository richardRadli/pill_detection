import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from tqdm import tqdm

from unet import UNet
from unet.data_loading import CustomDataset, BasicDataset
from utils.utils import create_timestamp, dice_coefficient, dice_loss, multiclass_dice_coefficient, use_gpu_if_available
from config.config import ConfigTrainingUnet
from config.const import DATA_PATH, IMAGES_PATH

cfg = ConfigTrainingUnet().parse()


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coefficient(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, \
                    'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coefficient(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)


def train_model(model, device, epochs: int = 5, batch_size: int = 1, learning_rate: float = 1e-5,
                val_percent: float = 0.1, save_checkpoint: bool = True, img_scale: float = 0.5, amp: bool = False,
                weight_decay: float = 1e-8, momentum: float = 0.999, gradient_clipping: float = 1.0):
    # 1. Create dataset
    try:
        dataset = CustomDataset(IMAGES_PATH.get_data_path("train_images"), IMAGES_PATH.get_data_path("train_masks"),
                                cfg.scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(IMAGES_PATH.get_data_path("train_images"), IMAGES_PATH.get_data_path("train_masks"),
                               cfg.scale)

    # 2. Split into train / validation partitions
    # Determine the ratio of the split
    train_ratio = 0.8

    # Determine the lengths of each split based on the ratio
    n_train = int(train_ratio * len(dataset))
    n_val = len(dataset) - n_train

    train_set, val_set = random_split(dataset, [n_train, n_val])

    # 3. Create data loaders
    loader_cfg = dict(batch_size=cfg.batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_cfg)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_cfg)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', dir=DATA_PATH.get_data_path("logs_unet"), resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if cfg.save_checkpoint:
            timestamp = create_timestamp()
            path_to_save = os.path.join(DATA_PATH.get_data_path("weights_unet"), timestamp)
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, (path_to_save + '/checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


if __name__ == '__main__':
    try:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        device = use_gpu_if_available()
        logging.info(f'Using device {device}')

        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        model = UNet(n_channels=3, n_classes=cfg.classes, bilinear=cfg.bilinear)

        logging.info(f'Network:\n'
                     f'\t{model.n_channels} input channels\n'
                     f'\t{model.n_classes} output channels (classes)\n'
                     f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

        if cfg.load:
            state_dict = torch.load(cfg.load, map_location=device)
            del state_dict['mask_values']
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {cfg.load}')

        model.to(device=device)
        summary(model, (3, 400, 400))

        try:
            train_model(
                model=model,
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.lr,
                device=device,
                img_scale=cfg.scale,
                val_percent=cfg.valid / 100,
                amp=cfg.amp
            )
        except torch.cuda.OutOfMemoryError:
            logging.error('Detected OutOfMemoryError! '
                          'Enabling checkpointing to reduce memory usage, but this slows down training. '
                          'Consider enabling AMP (--amp) for fast and memory efficient training')
            torch.cuda.empty_cache()
            model.use_checkpointing()
            train_model(
                model=model,
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.lr,
                device=device,
                img_scale=cfg.scale,
                val_percent=cfg.valid / 100,
                amp=cfg.amp
            )
    except KeyboardInterrupt as kie:
        print(kie)
