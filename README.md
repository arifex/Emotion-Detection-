# Facial Emotion Detection

Real-time facial emotion recognition from webcam or images.

## How it works
1. **Face detection:** OpenCV Haar cascade (`haarcascade_frontalface_default.xml`) locates faces.
2. **Preprocessing:** each face is cropped, converted to grayscale and resized to [48 × 48].
3. **Classification:** a CNN trained in Keras (`Emotion_Detection.h5`) predicts one of [7] emotions: [angry, disgust, fear, happy, neutral, sad, surprise].

## Dataset
[FER-2013 — ~35,000 grayscale 48 × 48 face images]

## Model
[e.g., 4 conv blocks (Conv–BatchNorm–ReLU–MaxPool–Dropout) → Dense → Softmax]
Loss: categorical cross-entropy · Optimizer: [Adam] · Epochs: [__]

## Results
Validation accuracy: [__ %]

## Usage
pip install tensorflow opencv-python numpy
python train.py   # train the model
python test.py    # run real-time detection with webcam

## Author
Md. Arifuzzaman — B.E. Computer Engineering, Gujarat Technological University
