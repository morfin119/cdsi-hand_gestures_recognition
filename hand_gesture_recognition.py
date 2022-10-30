# Machine Learning
from joblib import load
# Computer Vision
import cv2
import mediapipe as mp
# Image Manipulation
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
# Text to Speech
import pygame
import threading
from gtts import gTTS
# Virtual Keyboard
from pynput.keyboard import Key, Controller
# Misc
import numpy as np
import os
import pandas as pd
import time
# DIY Algorithms
import common

def say(text):
    clock = pygame.time.Clock()

    tts = gTTS(text, lang='es', tld='com')
    tts.save("sound.mp3")

    pygame.mixer.init()
    pygame.mixer.music.load('sound.mp3')

    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)
    pygame.mixer.music.stop()
    pygame.mixer.quit()

    os.remove("sound.mp3")

###
# LOAD MODEL
###
path = os.path.join('models', 'hand_gesture_model.joblib')
svclassifier = load(path)

###
# CV2 CONFIGURATIONS
###
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

###
# MEDIAPIPE CONFIGURATIONS
###
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

###
# PIL CONFIGURATIONS
###
font_path = os.path.join("resourses", "FreeSans.otf")
print(font_path)
font = ImageFont.truetype(font_path, 50)

###
# VIRTUAL KEYBOARD
###
keyboard = Controller()

###
# GET WIDTH AND HEIGHT OF IMAGE
###
_, img = cap.read()
h, w, _ = img.shape

###
# SET VARIABLES AND CONSTANTS TO CALCULATE FPS
###
prev_time = time.time()
# FPS update time in seconds
DISPLAY_TIME = 2
# Frame Count
fc = 0
# Frames per Second
fps = 0

###
# LETTER DETECTION VARIABLES
###
current_character = ''
last_character = ''
# Time in seconds a person has to make sign before playing sound
DETECTION_TIME = 1
# Previous detection time
prev_detection_time = time.time()
# Flag to block detection
detection_mode = True

###
# INIT THREAD
###
th = threading.Thread()

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)

    # Calculate FPS
    fc+=1
    time_diff = time.time() - prev_time
    if (time_diff) >= DISPLAY_TIME :
        fps = fc / (time_diff)
        fc = 0
        prev_time = time.time()
	
    # Add FPS count on frame
    fps_disp = f"FPS: {int(fps)}"
    cv2.putText(img, fps_disp, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    hand_landmarks = common.get_right_hand_landmarks(results, h, w)
    
    if hand_landmarks is False:
        detection_mode = True
    if hand_landmarks is not False:
        # Draw Bounding Box
        df = pd.DataFrame(hand_landmarks)
        xmin = df[0].min()
        xmax = df[0].max()
        ymin = df[1].min()
        ymax = df[1].max()
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 3)

        # Normalize landmarks
        normalized_landmarks = common.normalize_hand_landmarks(hand_landmarks)
        df = pd.DataFrame(normalized_landmarks)
        df.columns = ['x', 'y', 'z']
        df.drop('z', axis=1, inplace=True)
        features = df.to_numpy().flatten()

        # Print landmarks normlized values
        for index, landmark in enumerate(hand_landmarks):
            cv2.putText(img, f"x: {df.iloc[index]['x']:.2}, y:{df.iloc[index]['y']:.2}", 
                (landmark[0], landmark[1] + 12), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)

        # Make predictions
        last_character = current_character
        current_character = svclassifier.predict([features])
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 50),f'Prediction: {current_character[0]}',(255,0,255),font=font)
        img = np.asarray(img_pil)

        if (last_character != current_character):
            prev_detection_time = time.time()
            detection_mode = True

        # If current_character = last_character and time_diff > constant play sound
        if (last_character == current_character) \
                and (time.time() - prev_detection_time) > DETECTION_TIME \
                and detection_mode == True \
                and th.is_alive() == False:
            key = current_character[0]
            if key == 'null':
                continue
            th = threading.Thread(target=say, args=(key, ))
            th.start()
            keyboard.press(key)
            keyboard.release(key)
            keyboard.press(Key.enter)
            keyboard.release(Key.enter)
            detection_mode = False

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()
