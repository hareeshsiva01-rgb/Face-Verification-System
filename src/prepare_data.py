import cv2
import os
import numpy as np

# Location of our dataset
dataset_path = "dataset/raw"

# Lists to store images and labels
images = []
labels = []

# Go through each person's folder
for person_id in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_id)

    # Ignore files; only process folders
    if not os.path.isdir(person_folder):
        continue

    print("Processing:", person_id)

    # Read every image
    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            continue

        # Resize image
        image = cv2.resize(image, (160, 160))

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize pixel values
        image = image / 255.0

        # Add image
        images.append(image)

        # Add person ID
        labels.append(person_id)

# Convert to NumPy arrays
images = np.array(images, dtype=np.float32)
labels = np.array(labels)

print()
print("Data preparation completed.")
print("Number of images:", len(images))
print("Image shape:", images.shape)
print("Labels:", labels)