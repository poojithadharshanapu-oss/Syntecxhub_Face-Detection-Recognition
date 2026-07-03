import cv2
import os
import numpy as np

# Path to dataset
dataset_path = "dataset"

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_ids = {}
current_id = 0

# Read dataset
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    label_ids[current_id] = person_name

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        faces.append(image)
        labels.append(current_id)

    current_id += 1

# Train the recognizer
print("Training model...")
recognizer.train(faces, np.array(labels))

# Save trained model
os.makedirs("encodings", exist_ok=True)
recognizer.save("encodings/trainer.yml")

# Save label mapping
with open("encodings/labels.txt", "w") as f:
    for label, name in label_ids.items():
        f.write(f"{label},{name}\n")

print("Training completed successfully!")
print("Model saved as encodings/trainer.yml")
print("Labels saved as encodings/labels.txt")