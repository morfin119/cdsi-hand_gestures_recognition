import cv2
import numpy as np
from joblib import dump
from sklearn.svm import SVC
from pathlib import Path
import common
from tqdm import tqdm

# Define the path to the dataset directory using Path
dataset_directory_path = Path('..', 'data')

# Count the total number of files in the dataset directory
total_files = sum(1 for _ in dataset_directory_path.rglob('*.jpg'))

# Initialize tqdm progress bar
progress_bar = tqdm(total=total_files, desc='Processing dataset', unit='file')

# Lists to store labels and features
labels = []
features = []

# Iterate over each directory in the dataset
for label_directory_path in dataset_directory_path.iterdir():

    # Iterate over each file in the sign directory
    for instance_file_path in label_directory_path.iterdir():
        # Update progress bar
        progress_bar.update(1)

        # Read the image
        img = cv2.imread(str(instance_file_path), cv2.IMREAD_COLOR)
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image to extract hand landmarks using common function
        hand_landmarks = common.get_right_hand_landmarks(imgRGB)
        
        if hand_landmarks is not False:
            # Append the label to the label list
            labels.append(label_directory_path.name)

            # Normalize hand landmarks using common function
            normalized_landmarks = common.normalize_hand_landmarks(hand_landmarks)
            
            # Flatten the feature array
            flattened_features = np.array(normalized_landmarks).flatten()
            # Append the feature array to the instance list
            features.append(flattened_features) 

# Convert lists to numpy arrays
labels = np.asarray(labels)
features = np.asarray(features)

# Train a Support Vector Classifier
support_vector_classifier = SVC(kernel='poly', degree=3)
support_vector_classifier.fit(features, labels)

# Create a directory to save the model
models_directory = Path('..', 'models')
models_directory.mkdir(exist_ok=True)

# Define the path to save the model
model_file_path = models_directory / 'support_vector_classifier_model.joblib'

# Save the trained model
dump(support_vector_classifier, model_file_path)
