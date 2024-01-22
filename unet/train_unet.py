import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import wandb

from torch import optim
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config.config import ConfigTrainingUnet
from config.const import DATA_PATH, DATASET_PATH
from dataloader_unet import BasicDataset, UNetDataLoader
from loss_functions import multiclass_dice_coefficient, dice_coefficient, dice_loss
from utils.utils import measure_execution_time, use_gpu_if_available


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++ T R A I N   U N E T +++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TrainUnet:
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- __I N I T__ ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.cfg = ConfigTrainingUnet().parse()

        self.dir_img = DATASET_PATH.get_data_path("ogyei_v2_single_splitted_train_images")
        self.dir_mask = DATASET_PATH.get_data_path("ogyei_v2_single_splitted_gt_train_masks")
        self.dir_checkpoint = "C:/Users/ricsi/Documents/project/storage/IVM/data/unet/weights/"

        # Create dataset
        try:
            self.dataset = UNetDataLoader(str(self.dir_img), str(self.dir_mask), self.cfg.img_scale)
        except (AssertionError, RuntimeError, IndexError):
            self.dataset = BasicDataset(str(self.dir_img), str(self.dir_mask), self.cfg.img_scale)

        # Split into train / validation partitions
        self.n_val = int(len(self.dataset) * self.cfg.val)
        self.n_train = len(self.dataset) - self.n_val
        train_set, val_set = random_split(self.dataset, [self.n_train, self.n_val],
                                          generator=torch.Generator().manual_seed(0))

        # Create data loaders
        loader_args = dict(batch_size=self.cfg.batch_size, num_workers=1, pin_memory=True)
        self.train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        self.val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

        # Initialize logging
        self.experiment = wandb.init(project='U-Net',
                                     dir="C:/Users/ricsi/Documents/project/storage/IVM/data/unet/logs/",
                                     resume='allow',
                                     anonymous='must')
        self.experiment.config.update(
            dict(epochs=self.cfg.epochs, batch_size=self.cfg.batch_size, learning_rate=self.cfg.learning_rate,
                 val_percent=self.cfg.val, save_checkpoint=self.cfg.save_checkpoint, img_scale=self.cfg.img_scale,
                 amp=self.cfg.amp))

        self.device = use_gpu_if_available()

        # Define the UNet model
        self.model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=self.cfg.channels,
            classes=self.cfg.classes
        )

        # Modify the last layer for fine-tuning
        num_channels = self.model.segmentation_head[0].in_channels
        self.model.segmentation_head[0] = \
            nn.Conv2d(in_channels=num_channels, out_channels=self.cfg.classes, kernel_size=1).to(self.device)

        self.model = self.model.to(self.device)
        images = next(iter(self.train_loader))['image']
        height, width = images.shape[2:]
        summary(self.model, (self.cfg.channels, width, height))

        logging.info(f'Network:\n'
                     f'\t{self.cfg.channels} input channels\n'
                     f'\t{self.cfg.classes} output channels (classes)\n'
                     )

        # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=self.cfg.learning_rate,
                                       weight_decay=self.cfg.weight_decay,
                                       momentum=self.cfg.momentum)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=5)
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)
        self.criterion = nn.CrossEntropyLoss() if self.cfg.classes > 1 else nn.BCEWithLogitsLoss()
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
    def evaluate(self) -> float:
        """
        Evaluate the model on the validation set and compute the Dice score.

        This function sets the model to evaluation mode and computes the Dice score for each batch in the validation
        set. The Dice score is a common evaluation metric for semantic segmentation tasks. It measures the overlap
        between the predicted mask and the true mask.

        Returns:
            float: Average Dice score across the validation set.
        """

        self.model.eval()
        num_val_batches = len(self.val_loader)
        dice_score = 0

        # Iterate over the validation set
        with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.cfg.amp):
            for batch in tqdm(self.val_loader, total=num_val_batches, desc='Validation round', unit='batch',
                              leave=False):
                image, mask_true = batch['image'], batch['mask']

                # Move images and labels to correct device and type
                image = image.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=self.device, dtype=torch.long)

                # Predict the mask
                mask_pred = self.model(image)

                if self.cfg.classes == 1:
                    assert mask_true.min() >= 0 and mask_true.max() <= 1, \
                        'True mask indices should be in [0, 1]'
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

                    # Compute the Dice score
                    dice_score += dice_coefficient(mask_pred, mask_true, reduce_batch_first=False)
                else:
                    assert mask_true.min() >= 0 and mask_true.max() < self.cfg.classes, \
                        'True mask indices should be in [0, n_classes['

                    # Convert to one-hot format
                    mask_true = F.one_hot(mask_true, self.cfg.classes).permute(0, 3, 1, 2).float()
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), self.cfg.classes).permute(0, 3, 1, 2).float()

                    # Compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coefficient(mask_pred[:, 1:], mask_true[:, 1:],
                                                              reduce_batch_first=False)

        self.model.train()
        return dice_score / max(num_val_batches, 1)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- T R A I N ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @measure_execution_time
    def train(self) -> None:
        """
        Train the model.

        This function trains the model for the specified number of epochs.
        It iterates over the training data and performs forward and backward passes to update the model's parameters.
        After each epoch, it also performs model evaluation on the validation set and logs the results.

        The training process includes calculating the loss, optimizing the parameters using the optimizer,
        logging the training loss and step, and saving checkpoints.

        """

        # Begin training
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            epoch_loss = 0

            with tqdm(total=self.n_train, desc=f'Epoch {epoch}/{self.cfg.epochs}', unit='img') as pbar:
                for batch in self.train_loader:
                    images, true_masks = batch['image'], batch['mask']

                    assert images.shape[1] == self.cfg.channels, \
                        f'Network has been defined with {self.cfg.channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=self.device, dtype=torch.long)

                    with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.cfg.amp):
                        masks_pred = self.model(images)
                        if self.cfg.classes == 1:
                            loss = self.criterion(masks_pred.squeeze(1), true_masks.float())
                            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        else:
                            loss = self.criterion(masks_pred, true_masks)
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, self.cfg.classes).permute(0, 3, 1, 2).float(),
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
                                if value.grad is not None:
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

            # Save weights
            if self.cfg.save_checkpoint:
                state_dict = self.model.state_dict()
                state_dict['mask_values'] = self.dataset.mask_values
                torch.save(state_dict, os.path.join(self.dir_checkpoint, "epoch_" + str(epoch) + ".pt"))
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
