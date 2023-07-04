import cv2
import os

from tqdm import tqdm

# Path to the directory containing images
mask_dir = "C:/Users/ricsi/Desktop/cure_test/test_mask"

# Output directory to save annotations
output_dir = "C:/Users/ricsi/Desktop/cure_test/yolo_labels"

# Path to the directory containing images for visualization
image_dir = "C:/Users/ricsi/Desktop/cure_test/images"

for image_filename in tqdm(os.listdir(mask_dir), total=len(os.listdir(mask_dir))):
    mask_path = os.path.join(mask_dir, image_filename)
    if os.path.basename(mask_path).endswith('.png'):

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Find contours in the mask image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare the annotation file path
        annotation_filename = image_filename.replace(".png", ".txt")
        annotation_path = os.path.join(output_dir, annotation_filename)

        if os.path.basename(annotation_path).endswith('.txt'):
            print(annotation_path)
            with open(annotation_path, "w") as f:
                for contour in contours:
                    # Compute the bounding box coordinates
                    x, y, w, h = cv2.boundingRect(contour)

                    # Normalize the bounding box coordinates
                    img_width, img_height = mask.shape[1], mask.shape[0]
                    x_norm = x / img_width
                    y_norm = y / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height

                    # Determine the class label (extract from the image filename)
                    class_label = image_filename.split("_")[0]

                    # Write the annotation to the file
                    line = f"{class_label} {x_norm} {y_norm} {w_norm} {h_norm}\n"
                    print(line)
                    f.write(line)

            # Load the corresponding image
            image_path = os.path.join(image_dir, image_filename)
            image = cv2.imread(image_path)

            # Convert YOLO annotation to bounding box coordinates
            with open(annotation_path, "r") as f:
                for line in f:
                    class_label, x_norm, y_norm, w_norm, h_norm = line.strip().split()

                    # Convert normalized coordinates to pixel coordinates
                    x = int(float(x_norm) * image.shape[1])
                    y = int(float(y_norm) * image.shape[0])
                    w = int(float(w_norm) * image.shape[1])
                    h = int(float(h_norm) * image.shape[0])

                    # Draw the bounding box on the image
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the image with bounding boxes
            cv2.imshow("Image with Bounding Boxes", cv2.resize(image, (640, 640)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
