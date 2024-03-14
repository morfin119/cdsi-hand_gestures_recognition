from cvzone.HandTrackingModule import HandDetector
from sklearn.preprocessing import MinMaxScaler

# Create a HandDetector object outside the function
detector = HandDetector(detectionCon=0.5, maxHands=1)

def get_right_hand_landmarks(frame):
    """
    Detects and extracts landmarks of the right hand from the given frame using cvzone.

    Args:
        frame (numpy.ndarray): The input frame containing the image.

    Returns:
        list of tuple or bool: If the right hand is detected, returns a list of tuples containing (x, y) coordinates 
                               of landmarks. If no right hand is detected, returns False.
    """
    # Detect hands in the frame using the previously created detector
    hands, _ = detector.findHands(frame)
    
    # Check if any hands are detected
    if hands:
        # Iterate through each detected hand
        for hand in hands:
            # Check if the detected hand is the right hand
            #if hand["type"] == "Right":
            landmarks = []
            # Iterate through landmarks of the right hand
            for lm in hand["lmList"]:
                # Extract x, y coordinates of the landmark
                cx, cy = lm[1], lm[2]
                landmarks.append((cx, cy))
            return landmarks  # Return the landmarks of the right hand
        
    return False  # Return False if no right hand is detected
    
def normalize_hand_landmarks(landmarks):
    """
    Normalize hand landmarks using Min-Max scaling.

    Args:
        landmarks (list of tuples): List of landmark coordinates (x, y).

    Returns:
        list of tuples: Normalized landmark coordinates.
    """
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    
    # Fit the scaler to the data and perform normalization
    normalized_landmarks = scaler.fit_transform(landmarks)
    
    # Convert the normalized array back to a list of tuples
    normalized_landmarks = [tuple(row) for row in normalized_landmarks]

    return normalized_landmarks
