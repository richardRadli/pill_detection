import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess the image
image_path = "C:/Users/ricsi/Documents/project/storage/IVM/datasets/ogyi_multi/splitted/test/images/" \
             "id_mul_001_002_003_ambroxol_egis-dorithricin_mentol-cataflam_v_011.png"
input_image = Image.open(image_path).convert("RGB")

preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to torch.Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

input_tensor = preprocess(input_image).unsqueeze(0)

# Ensure the image tensor is on the same device as the model
if torch.cuda.is_available():
    input_tensor = input_tensor.to("cuda")
    model.to("cuda")

# Make predictions on the input image
with torch.no_grad():
    predictions = model(input_tensor)

# Access the predicted bounding boxes and labels
boxes = predictions[0]['boxes'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()
masks = predictions[0]['masks'].cpu().numpy()

# Visualize the image with bounding boxes
input_image_np = np.array(input_image)
plt.imshow(input_image_np)

# Plot the bounding boxes on the image
for i in range(len(boxes)):
    box = boxes[i]
    label = labels[i]
    mask = masks[i, 0]
    mask_denormalized = (mask * 255).astype(np.uint8)
    cv2.imshow("mask", cv2.resize(mask_denormalized, (2465//3, 1683//3)))
    cv2.waitKey()

    # Plot the bounding box
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r',
                             facecolor='none')
    plt.gca().add_patch(rect)

    # Add label text
    plt.text(box[0], box[1], str(label), color='w', fontsize=8, verticalalignment='top', bbox={'color': 'r', 'pad': 0})

plt.show()

# # Plot each bounding box as an individual image
# for i in range(len(boxes)):
#     box = boxes[i]
#     label = labels[i]
#
#     # Extract the bounding box region
#     xmin, ymin, xmax, ymax = box
#     region = input_image.crop((xmin, ymin, xmax, ymax))
#     print(region)
#
#     # Plot the bounding box region as an individual image
#     plt.imshow(region)
#     plt.axis('off')
#     plt.title(f"Label: {label}")
#     plt.show()
#
