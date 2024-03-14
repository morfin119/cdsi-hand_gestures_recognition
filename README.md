# Mexican Sing Language Static Recognition using Support Vector Machines

This project implements recognition of static signs from the Mexican Sign Language (LSM) alphabet using Support Vector Machines (SVM) and computer vision techniques. It detects hand landmarks using the cvzone library and classifies them into the predefined signs of the LSM alphabet using an SVM classifier.

## Description

The code in this repository performs the following tasks:
- Reads images from a dataset directory
- Detects hand landmarks using cvzone
- Normalizes hand landmarks
- Trains an SVM classifier to recognize hand gestures
- Saves the trained model for later use

## Installation

To run this code, ensure you have Python installed on your system. Clone this repository and navigate to the project directory. Then, install the required dependencies using pip:

```
pip install -r requirements.txt
```

## Usage

1. First, create the models by executing the `create_models.py` script. This script uses a small dataset included in the repository to train the SVM classifier.

2. After creating the models, you can execute the main recognizer `hand_gesture_recognition.py` to perform real-time hand gesture recognition.


Ensure that you have a webcam connected to your system to capture real-time video for gesture recognition.

## License

This project is licensed under the [MIT License](LICENSE).
