import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import cv2
import torchvision.models.segmentation
import torch

imageSize = [2465, 1683]
imgPath = "C:/Users/ricsi/Desktop/ogyei_v2/train/images/id_001_ambroxol_egis_30mg_001.png"

device = "cpu"
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
model.load_state_dict(torch.load("C:/Users/ricsi/Desktop/2.torch"))
model.to(device)
model.eval()

images = cv2.imread(imgPath)
images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)
images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
images = images.swapaxes(1, 3).swapaxes(2, 3)
images = list(image.to(device) for image in images)

with torch.no_grad():
    pred = model(images)
    print(pred)

im = images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)
im2 = im.copy()
for i in range(len(pred[0]['masks'])):
    msk = pred[0]['masks'][i, 0].detach().cpu().numpy()
    scr = pred[0]['scores'][i].detach().cpu().numpy()
    if scr > 0.8:
        im2[:, :, 0][msk > 0.5] = random.randint(0, 255)
        im2[:, :, 1][msk > 0.5] = random.randint(0, 255)
        im2[:, :, 2][msk > 0.5] = random.randint(0, 255)

    cv2.imshow(str(scr), np.hstack([im, im2]))
    cv2.waitKey()
