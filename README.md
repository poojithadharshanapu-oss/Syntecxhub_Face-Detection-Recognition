# Face Detection & Recognition using OpenCV (LBPH)

## Overview
This project detects and recognizes human faces in real time using OpenCV. It uses Haar Cascade for face detection and the LBPH Face Recognizer from OpenCV Contrib for face recognition.

## Features
- Face detection using Haar Cascade
- Face recognition using LBPH Face Recognizer
- Supports multiple faces
- Displays bounding boxes and names
- Register new people using a webcam
- Train the recognizer using captured images

## Technologies Used
- Python
- OpenCV
- OpenCV Contrib
- NumPy

## Project Structure

```
Face_Detection_Recognition/
├── screenshots/
├── register_face.py
├── train_model.py
├── recognize_faces.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Register a new face

```bash
python register_face.py
```

### 3. Train the model

```bash
python train_model.py
```

### 4. Start recognition

```bash
python recognize_faces.py
```

## Output
- Detects faces from the webcam
- Draws bounding boxes around faces
- Displays the recognized person's name
- Labels unknown faces as "Unknown"

## Author
Poojitha
