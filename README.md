# Real-Time Facial Emotion Detection

Real-time facial emotion recognition from a webcam. OpenCV finds faces in each frame and a Keras deep learning model classifies each face into one of **5 emotions**: Angry, Happy, Neutral, Sad, Surprise.

Undergraduate project, B.E. Computer Engineering, Gujarat Technological University (2019).

## How it works

```
Webcam frame → grayscale → Haar-cascade face detection → crop each face
            → resize → CNN classifier → emotion label drawn on the frame
```

1. **Face detection:** OpenCV Haar cascade (`haarcascade_frontalface_default.xml`)
2. **Emotion classification:** trained Keras model (`Emotion_Detection.h5`)
3. **Display:** bounding box and predicted emotion shown live; press **`q`** to quit

## Training (`train.py`)

| Setting | Value |
|---|---|
| Dataset | FER-2013 (5-class subset: Angry, Happy, Neutral, Sad, Surprise) |
| Images | 24,176 training / 3,006 validation |
| Model | ImageNet-pretrained **MobileNet** (transfer learning) + GlobalAveragePooling → Dense 1024 → Dense 1024 → Dense 512 → Dense 5 (softmax) |
| Input | 224 × 224 × 3 |
| Augmentation | `ImageDataGenerator` (rescale, rotation, shifts, zoom, horizontal flip) |
| Optimizer / loss | Adam (lr = 0.001), categorical cross-entropy |
| Training | 25 epochs, batch size 32 |
| Callbacks | EarlyStopping (patience 10, restore best weights), ModelCheckpoint, ReduceLROnPlateau |

## Dataset

**FER-2013** (Goodfellow et al., 2013): 48 × 48 grey-scale face images labelled with 7 emotions, of which this project uses 5. Download it from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and arrange it as:

```
fer2013/
├── train/       Angry/ Happy/ Neutral/ Sad/ Surprise/
└── validation/  Angry/ Happy/ Neutral/ Sad/ Surprise/
```

The dataset is not included in this repository.

## Usage

```bash
git clone https://github.com/arifex/Emotion-Detection-.git
cd Emotion-Detection-
pip install tensorflow keras opencv-python numpy
```

**Run real-time detection with the trained model:**
```bash
python test.py
```

**Retrain the model:** set the dataset paths at the top of `train.py`, then run
```bash
python train.py
```

## Repository structure

```
├── train.py                              # model training (MobileNet transfer learning)
├── test.py                               # real-time webcam emotion detection
├── Emotion_Detection.h5                  # trained model
└── haarcascade_frontalface_default.xml   # OpenCV face detector
```

## Tech stack

Python · TensorFlow / Keras · OpenCV · NumPy

## Author

**Md. Arifuzzaman** — Assistant Professor, Department of CSE, Leading University, Sylhet, Bangladesh
[GitHub](https://github.com/arifex) · [Google Scholar](https://scholar.google.com/citations?hl=en&user=ngntKZ8AAAAJ)
