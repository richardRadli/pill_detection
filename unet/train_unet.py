import logging
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch

import wandb
from unet_model import UNet
from data_loader_unet import BasicDataset, UNetDataLoader
from config.config import ConfigTrainingUnet
from utils.utils import multiclass_dice_coefficient, dice_coefficient, dice_loss


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++ T R A I N   U N E T +++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TrainUnet:
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.cfg = ConfigTrainingUnet().parse()

        self.dir_img = Path('C:/Users/ricsi/Desktop/train/images/')
        self.dir_mask = Path('C:/Users/ricsi/Desktop/train/masks/')
        self.dir_checkpoint = Path('C:/Users/ricsi/Desktop/checkpoints/')

        # 1. Create dataset
        try:
            self.dataset = UNetDataLoader(str(self.dir_img), str(self.dir_mask), self.cfg.img_scale)
        except (AssertionError, RuntimeError, IndexError):
            self.dataset = BasicDataset(str(self.dir_img), str(self.dir_mask), self.cfg.img_scale)

        # 2. Split into train / validation partitions
        val_percent = self.cfg.val / 100
        self.n_val = int(len(self.dataset) * val_percent)
        self.n_train = len(self.dataset) - self.n_val
        train_set, val_set = random_split(self.dataset, [self.n_train, self.n_val],
                                          generator=torch.Generator().manual_seed(0))

        # 3. Create data loaders
        loader_args = dict(batch_size=self.cfg.batch_size, num_workers=1, pin_memory=True)
        self.train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        self.val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

        # (Initialize logging)
        self.experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
        self.experiment.config.update(
            dict(epochs=self.cfg.epochs, batch_size=self.cfg.batch_size, learning_rate=self.cfg.learning_rate,
                 val_percent=val_percent, save_checkpoint=self.cfg.save_checkpoint, img_scale=self.cfg.img_scale,
                 amp=self.cfg.amp)
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')

        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        self.model = UNet(n_channels=3, n_classes=self.cfg.classes, bilinear=self.cfg.bilinear)
        self.model = self.model.to(self.device)

        logging.info(f'Network:\n'
                     f'\t{self.model.n_channels} input channels\n'
                     f'\t{self.model.n_classes} output channels (classes)\n'
                     f'\t{"Bilinear" if self.model.bilinear else "Transposed conv"} upscaling')

        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay,
                                       momentum=self.cfg.momentum)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=5)
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)
        self.criterion = nn.CrossEntropyLoss() if self.model.n_classes > 1 else nn.BCEWithLogitsLoss()
        self.global_step = 0

        logging.info(f'''Starting training:
                    Epochs:          {self.cfg.epochs}
                    Batch size:      {self.cfg.batch_size}
                    Learning rate:   {self.cfg.learning_rate}
                    Training size:   {self.n_train}
                    Validation size: {self.n_val}
                    Checkpoints:     {self.cfg.save_checkpoint}
                    Device:          {self.device}
                    Images scaling:  {self.cfg.img_scale}
                    Mixed Precision: {self.cfg.amp}
                ''')

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ E V A L U A T E -------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()
        num_val_batches = len(self.val_loader)
        dice_score = 0

        # iterate over the validation set
        with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.cfg.amp):
            for batch in tqdm(self.val_loader, total=num_val_batches, desc='Validation round', unit='batch',
                              leave=False):
                image, mask_true = batch['image'], batch['mask']

                # move images and labels to correct device and type
                image = image.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=self.device, dtype=torch.long)

                # predict the mask
                mask_pred = self.model(image)

                if self.model.n_classes == 1:
                    assert mask_true.min() >= 0 and mask_true.max() <= 1, \
                        'True mask indices should be in [0, 1]'
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                    # compute the Dice score
                    dice_score += dice_coefficient(mask_pred, mask_true, reduce_batch_first=False)
                else:
                    assert mask_true.min() >= 0 and mask_true.max() < self.model.n_classes, \
                        'True mask indices should be in [0, n_classes['
                    # convert to one-hot format
                    mask_true = F.one_hot(mask_true, self.model.n_classes).permute(0, 3, 1, 2).float()
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), self.model.n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coefficient(mask_pred[:, 1:], mask_true[:, 1:],
                                                              reduce_batch_first=False)

        self.model.train()
        return dice_score / max(num_val_batches, 1)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- T R A I N ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def train(self):
        # 5. Begin training
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            epoch_loss = 0
            with tqdm(total=self.n_train, desc=f'Epoch {epoch}/{self.cfg.epochs}', unit='img') as pbar:
                for batch in self.train_loader:
                    images, true_masks = batch['image'], batch['mask']

                    assert images.shape[1] == self.model.n_channels, \
                        f'Network has been defined with {self.model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=self.device, dtype=torch.long)

                    with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.cfg.amp):
                        masks_pred = self.model(images)
                        if self.model.n_classes == 1:
                            loss = self.criterion(masks_pred.squeeze(1), true_masks.float())
                            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        else:
                            loss = self.criterion(masks_pred, true_masks)
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, self.model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )

                    self.optimizer.zero_grad(set_to_none=True)
                    self.grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clipping)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

                    pbar.update(images.shape[0])
                    self.global_step += 1
                    epoch_loss += loss.item()
                    self.experiment.log({
                        'train loss': loss.item(),
                        'step': self.global_step,
                        'epoch': epoch
                    })
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # Evaluation round
                    division_step = (self.n_train // (5 * self.cfg.batch_size))
                    if division_step > 0:
                        if self.global_step % division_step == 0:
                            histograms = {}
                            for tag, value in self.model.named_parameters():
                                tag = tag.replace('/', '.')
                                if not (torch.isinf(value) | torch.isnan(value)).any():
                                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                            val_score = self.evaluate()
                            self.scheduler.step(val_score)

                            logging.info('Validation Dice score: {}'.format(val_score))

                            self.experiment.log({
                                'learning rate': self.optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': self.global_step,
                                'epoch': epoch,
                                **histograms
                            })

            if self.cfg.save_checkpoint:
                Path(self.dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = self.model.state_dict()
                state_dict['mask_values'] = self.dataset.mask_values
                torch.save(state_dict, str(self.dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- M A I N ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    train_unet = TrainUnet()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        train_unet.train()
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        train_unet.model.use_checkpointing()
        train_unet.train()
