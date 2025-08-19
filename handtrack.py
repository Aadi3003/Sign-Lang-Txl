print ("Hello World!") # This is just to test that the file is executing.

# It's venv python 3.11.7 that I'm runnig, cuz mediapipe is not available for higher vers.
# Cataloguing my progress and setup, again, tis just a prototype for now.

# press Q to exit the window after running it.

# Before running, check if (venv) is there in terminal before line, if not:
# .\venv\Scripts\Activate.ps1
# Or, just run it first then can work on it.

# test change

import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# prediction
import pandas as pd

# Load saved dataset
df = pd.read_csv("hand_dataset.csv")

print("Dataset loaded:", df.shape)
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Split features (X) and labels (y)
X = df.drop("label", axis=1).values
y = df["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple KNN classifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

print("Classifier trained, accuracy on test:", clf.score(X_test, y_test))
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Split features (X) and labels (y)
X = df.drop("label", axis=1).values
y = df["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple KNN classifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

print("Classifier trained, accuracy on test:", clf.score(X_test, y_test))

# === Setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATASET_FILE = "hand_dataset.csv"

# Create CSV file with header if not exists
if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"] + [f"f{i}" for i in range(126)]
        writer.writerow(header)

def landmarks_to_np(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

def normalize_hand(landmarks21):
    wrist = landmarks21[0].copy()
    translated = landmarks21 - wrist
    ref_dist = np.linalg.norm(landmarks21[0] - landmarks21[9])
    if ref_dist < 1e-6:
        ref_dist = 1.0
    scaled = translated / ref_dist
    return scaled.flatten()

def features_from_result(result):
    left_feat = np.zeros(63, dtype=np.float32)
    right_feat = np.zeros(63, dtype=np.float32)

    if not result.multi_hand_landmarks:
        return np.concatenate([left_feat, right_feat])

    for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
        raw_label = handedness.classification[0].label
        label = "Right" if raw_label == "Left" else "Left"  # flip fix
        lm = landmarks_to_np(hand_landmarks)
        feat = normalize_hand(lm)
        if label == "Left":
            left_feat = feat
        else:
            right_feat = feat
    return np.concatenate([left_feat, right_feat])

def draw_info(frame, feature_vec):
    text = f"Features len={len(feature_vec)} | Press 0-9 to save, q to quit"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,255,255), 2, cv2.LINE_AA)
    return frame

# === Main loop ===
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        feat_vec = features_from_result(result)
        frame = draw_info(frame, feat_vec)

        # Only predict if we have a non-zero feature vector
        if np.any(feat_vec):
            pred = clf.predict([feat_vec])[0]
            cv2.putText(frame, f"Predicted: {pred}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow("Hand Tracking + Data Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):   # quit
            break
        elif ord('0') <= key <= ord('9'):  # numbers 0-9
            label = chr(key)
            with open(DATASET_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                row = [label] + feat_vec.tolist()
                writer.writerow(row)
            print(f"[SAVED] Gesture {label}")

cap.release()
cv2.destroyAllWindows()