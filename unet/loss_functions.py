import torch

from torch import Tensor


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------  D I C E   C O E F F ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def dice_coefficient(input_tensor: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) \
        -> Tensor:
    """
    Average of Dice coefficient for all batches, or for a single mask

    :param input_tensor: The input tensor with shape (batch_size, num_classes, height, width).
    :param target: The target tensor with shape (batch_size, num_classes, height, width)
    :param reduce_batch_first: A flag indicating whether to reduce the batch dimension first before computing the
    Dice coefficient. Default is False.
    :param epsilon: A small value to prevent division by zero. Default is 1e-6.
    :return: The Dice coefficient computed for each batch or for a single mask, depending on the reduce_batch_first
    flag. If reduce_batch_first is True, the returned tensor has shape (num_classes,).
    If reduce_batch_first is False, the returned tensor has shape (batch_size, num_classes)
    """

    assert input_tensor.size() == target.size()
    assert input_tensor.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input_tensor.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input_tensor * target).sum(dim=sum_dim)
    sets_sum = input_tensor.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ M C L A S S   D I C E   C O E F F -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def multiclass_dice_coefficient(input_tensor: Tensor, target: Tensor, reduce_batch_first: bool = False,
                                epsilon: float = 1e-6):
    """
    Calculate the average Dice coefficient for all classes in a multi-class segmentation task.

    :param input_tensor: The input tensor with shape [batch_size, num_classes, height, width]
    :param target: The target tensor with shape [batch_size, num_classes, height, width].
    :param reduce_batch_first: Whether to reduce the batch size before calculating the Dice coefficient.
    :param epsilon: A small value added to the denominator to avoid division by zero.
    :return: The average Dice coefficient for all classes.
    """

    return dice_coefficient(input_tensor.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ D I C E   L O S S ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def dice_loss(input_tensor: Tensor, target: Tensor, multiclass: bool = False) -> float:
    """
    Dice loss (objective to minimize) between 0 and 1

    :param input_tensor: a tensor representing the predicted mask, of shape (batch_size, num_classes, height, width)
    :param target: a tensor representing the ground truth mask, of shape (batch_size, num_classes, height, width)
    :param multiclass: a boolean flag that indicates whether the input and target tensors represent a multiclass
    segmentation task, where each pixel can belong to one of several classes
    :return: a scalar tensor representing the Dice loss between the input and target tensors, which is a value between
    0 and 1.
    """

    fn = multiclass_dice_coefficient if multiclass else dice_coefficient
    return 1 - fn(input_tensor, target, reduce_batch_first=True)
