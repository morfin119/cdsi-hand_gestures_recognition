import common
import cv2
import os
import pandas as pd
import numpy as np
import mediapipe as mp
import sys
from joblib import dump
from sklearn.svm import SVC

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)

dataset_path = 'hand_dataset'

y = []
X = []

for directory in os.listdir(dataset_path):
    sign_path = os.path.join(dataset_path, directory)
    sys.stdout.write("[INFO] Processing %s\r" % (sign_path))
    sys.stdout.flush()
    
    for file in os.listdir(sign_path):
        file_path = os.path.join(sign_path, file)

        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        h, w, c = img.shape
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        hand_landmarks = common.get_right_hand_landmarks(results, h, w)
        if hand_landmarks is not False:
            y.append(directory)
            normalized_landmarks = common.normalize_hand_landmarks(hand_landmarks)
            df = pd.DataFrame(normalized_landmarks)
            df.columns = ['x', 'y', 'z']
            df.drop('z', axis=1, inplace=True)
            features = df.to_numpy().flatten()
            X.append(features)

y = np.asarray(y)
X = np.asarray(X)

svclassifier = SVC(kernel='poly', degree=3)
svclassifier.fit(X, y)

path = os.path.join('models', 'hand_gesture_model.joblib')
dump(svclassifier, path)
