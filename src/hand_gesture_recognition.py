from joblib import load
import cv2
import numpy as np
from pathlib import Path
import time
import common

# Load the model
model_file_path = Path('..', 'models', 'support_vector_classifier_model.joblib')
support_vector_classifier = load(str(model_file_path))

# Configure cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables and constants for FPS calculation
prev_time = time.time()
frame_count = 0
fps = 0
# FPS update time in seconds
DISPLAY_TIME = 2

while True:
    # Read frame from webcam
    _, img = cap.read()
    img = cv2.flip(img, 1)

    # Calculate FPS
    frame_count += 1
    time_diff = time.time() - prev_time
    if time_diff >= DISPLAY_TIME:
        fps = frame_count / time_diff
        frame_count = 0
        prev_time = time.time()
	
    # Add FPS count on frame
    cv2.putText(img, f"FPS: {int(fps)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)

    # Process the image to extract hand landmarks using common function
    hand_landmarks = common.get_right_hand_landmarks(img)

    if hand_landmarks is not False:
        # Normalize hand landmarks using common function
        normalized_landmarks = common.normalize_hand_landmarks(hand_landmarks)
            
        # Flatten the feature array
        features = np.array(normalized_landmarks).flatten()

        # Predict
        current_character = support_vector_classifier.predict([features])

        # Draw text on the image
        cv2.putText(img, f'Prediction: {current_character[0]}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 255), 2)
        
    # Add exit instruction
    cv2.putText(img, "Press 'q' to exit", (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display image
    cv2.imshow("Image", img)

    # Check for quit key
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
cap.release()
