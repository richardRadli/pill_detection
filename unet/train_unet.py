import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.nn.functional as func
import wandb

from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from tqdm import tqdm

from unet import StackedUNet
from unet.data_loading import CustomDataset, BasicDataset
from utils.utils import dice_loss, multiclass_dice_coefficient, dice_coefficient, create_timestamp
from config import ConfigTrainingUnet
from const import CONST

cfg = ConfigTrainingUnet().parse()


class TrainUNET:
    def __init__(self):
        # 1. Create dataset
        try:
            self.dataset = CustomDataset(CONST.dir_train_images, CONST.dir_train_masks, cfg.scale)
        except (AssertionError, RuntimeError, IndexError):
            self.dataset = BasicDataset(CONST.dir_train_images, CONST.dir_train_masks, cfg.scale)

        # # 2. Split into train / validation partitions
        # Determine the ratio of the split
        train_ratio = 0.8

        # Determine the lengths of each split based on the ratio
        self.n_train = int(train_ratio * len(self.dataset))
        self.n_val = len(self.dataset) - self.n_train

        train_set, val_set = random_split(self.dataset, [self.n_train, self.n_val])

        # # 3. Create data loaders
        loader_cfg = dict(batch_size=cfg.batch_size, num_workers=1, pin_memory=True)
        self.train_loader = DataLoader(train_set, shuffle=True, **loader_cfg)
        self.val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_cfg)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- E V A L U A T E ------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def evaluate(self, net, dataloader, device, amp):
        """
        This is a function that evaluates a neural network model on a validation dataset. The function Sets the neural
        network to evaluation mode, iterate over the validation set, and for each batch, Move the batch of images and
        labels to the specified device and data type. Predict the output masks for the images using the neural network
        model. Compute the Dice score between the predicted masks and the true masks for the batch. If the model has
        more than one class, the Dice score is computed for each class separately. Set the neural network back to
        training mode.

        :param net: the neural network model to be evaluated
        :param dataloader: a PyTorch DataLoader object that contains the validation datas
        :param device: the device (CPU or GPU) to use for the evaluation
        :param amp: a boolean flag indicating whether to use automatic mixed precision (AMP) for faster and more
        memory-efficient computation on certain GPUs
        :return: the average Dice score across all batches in the validation set
        """

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
                    mask_pred = (func.sigmoid(mask_pred) > 0.5).float()
                    # compute the Dice score
                    dice_score += dice_coefficient(mask_pred, mask_true, reduce_batch_first=False)
                else:
                    assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, \
                        'True mask indices should be in [0, n_classes['
                    # convert to one-hot format
                    mask_true = func.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                    mask_pred = func.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coefficient(mask_pred[:, 1:], mask_true[:, 1:],
                                                              reduce_batch_first=False)

        net.train()

        return dice_score / max(num_val_batches, 1)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- T R A I N   M O D E L ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def train(self, model, device, timestamp):
        """
        The function begins by creating a dataset using either a custom dataset class or a basic dataset class,
        depending on whether certain assertions are true. Next, the function initializes the logging, which involves
        setting up a W&B experiment and logging various information about the training process.

        :param model: the UNet model instance
        :param device: the device (CPU or GPU) to use for training
        :param timestamp:
        """

        # Initialize logging
        experiment = wandb.init(project='U-Net', dir=CONST.dir_wandb_checkpoint_logs, resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=cfg.epochs, batch_size=cfg.batch_size, learning_rate=cfg.lr,
                 val_percent=cfg.valid, save_checkpoint=cfg.save_checkpoint, img_scale=cfg.scale, amp=cfg.amp)
        )

        optimizer = optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  momentum=cfg.momentum)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
        global_step = 0

        # Beginning of the training
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            epoch_loss = 0
            with tqdm(total=self.n_train, desc=f'Epoch {epoch}/{cfg.epochs}', unit='img') as pbar:
                for batch in self.train_loader:
                    images, true_masks = batch['image'], batch['mask']

                    assert images.shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=cfg.amp):
                        masks_pred = model(images)
                        if model.n_classes == 1:
                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                            loss += dice_loss(fun.sigmoid(masks_pred.squeeze(1)), true_masks.float(),
                                              multiclass=False)
                        else:
                            loss = criterion(masks_pred, true_masks)
                            loss += dice_loss(
                                fun.softmax(masks_pred, dim=1).float(),
                                fun.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clipping)
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
                    division_step = (self.n_train // (5 * cfg.batch_size))
                    if division_step > 0:
                        if global_step % division_step == 0:
                            histograms = {}
                            for tag, value in model.named_parameters():
                                tag = tag.replace('/', '.')
                                if not torch.isinf(value).any():
                                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                if not torch.isinf(value.grad).any():
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                            val_score = self.evaluate(model, self.val_loader, device, cfg.amp)
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
                            except Exception as e:
                                print(e)

            if cfg.save_checkpoint:
                path_to_save = os.path.join(CONST.dir_checkpoint, timestamp)
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = self.dataset.mask_values
                torch.save(state_dict, (path_to_save + '/checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------- M A I N --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def main(self):
        """
        This function of a program that trains a UNet model for image segmentation.

        1. The function sets up logging to display information about the progress of the program during training.
        2.The function checks if CUDA is available on the system and sets the device to use either CUDA or CPU accord.
        3.The function creates an instance of the UNet model, passing in the number of input channels, output channels,
        and upscaling method (bilinear or transposed convolution).
        4.If a pre-trained model is specified in the configuration file, the function loads the model weights from the
        specified file.
        5. The function moves the model to the device (CPU or GPU) specified earlier.

        :return:
        """
        timestamp = create_timestamp()

        logging.basicConfig(level=logging.INFO)
        logging.info(f'''Starting training:
                Epochs:          {cfg.epochs}
                Batch size:      {cfg.batch_size}
                Learning rate:   {cfg.lr}
                Training size:   {self.n_train}
                Validation size: {self.n_val}
                Checkpoints:     {cfg.save_checkpoint}
                Device:          {self.device.type}
                Images scaling:  {cfg.scale}
                Mixed Precision: {cfg.amp}
            ''')
        logging.info(f'Using device {self.device}')

        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        model = StackedUNet(3, cfg.classes, bilinear=cfg.bilinear)

        logging.info(f'Network:\n'
                     f'\t{model.n_channels} input channels\n'
                     f'\t{model.n_classes} output channels (classes)\n'
                     f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

        if cfg.load:
            state_dict = torch.load(cfg.load, map_location=self.device)
            del state_dict['mask_values']
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {cfg.load}')

        print("\nCuda available: ", torch.cuda.is_available(), "\nNumber of devices: ", torch.cuda.device_count(),
              "\nCurrent device: ", torch.cuda.current_device())
        model.to(device=self.device)

        summary(model, (3, 400, 400))

        self.train(model=model, timestamp=timestamp, device=self.device)


if __name__ == '__main__':
    try:
        train_unet = TrainUNET()
        train_unet.main()
    except KeyboardInterrupt as kie:
        print(kie)
