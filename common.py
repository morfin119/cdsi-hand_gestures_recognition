from zmq import Message
import mediapipe as mp
import cv2
import pandas as pd

from google.protobuf.json_format import MessageToDict

def get_right_hand_landmarks(mp_solution_output, h, w):
    if mp_solution_output.multi_hand_landmarks is not None:
        for index, hand_handedness in enumerate(mp_solution_output.multi_handedness):
            handedness_dict = MessageToDict(hand_handedness)
            whichHand = (handedness_dict['classification'][0]['label'])
            if whichHand == "Right":
                right_hand_landmarks_dict = MessageToDict(mp_solution_output.multi_hand_landmarks[index])
                landmarks = []
                for lm in right_hand_landmarks_dict['landmark']:
                    cx, cy, cz = int(lm['x']*w), int(lm['y']*h), lm['z']
                    landmarks.append((cx, cy, cz))
                return landmarks
    return False
    
def normalize_hand_landmarks(landmarks):
    df = pd.DataFrame(landmarks)

    xmin = df[0].min()
    xmax = df[0].max()
    ymin = df[1].min()
    ymax = df[1].max()
    zmin = df[2].min()
    zmax = df[2].max()

    normalized_landmarks = []
    for _, row in df.iterrows():
        cx = row[0]
        cy = row[1]
        cz = row[2]
        nx = (cx - xmin) / (xmax - xmin)
        ny = (cy - ymin) / (ymax - ymin)
        nz = (cz - zmin) / (zmax - zmin)
        normalized_landmarks.append((nx, ny, nz))

    return normalized_landmarks
