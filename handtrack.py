# DON'T GIT PUSH IT YET!!! WORK IN PROGRESS!!! ONLY DO WHEN IT'S WORKING AS INTENDED!!!

# Initial print is just a check that the file is executing.
# It's venv python 3.11.7 that I'm runnig, cuz mediapipe is not available for higher vers.
# Before running, check if (venv) is in terminal before execution line, if not, try running it, else:
# .\venv\Scripts\Activate.ps1

# test change

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# === Files ===
STATIC_FILE = "hand_dataset.csv"
DYNAMIC_FILE = "dynamic_dataset.csv"

clf_static, clf_dynamic = None, None

# === Load static dataset ===
if os.path.exists(STATIC_FILE):
    df = pd.read_csv(STATIC_FILE)
    if len(df) > 0:
        X = df.drop("label", axis=1).values
        y = df["label"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf_static = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
        print("Static classifier trained, acc:", clf_static.score(X_test, y_test))

# === Load dynamic dataset ===
if os.path.exists(DYNAMIC_FILE):
    df_dyn = pd.read_csv(DYNAMIC_FILE)
    if len(df_dyn) > 0:
        Xd = df_dyn.drop("label", axis=1).values
        yd = df_dyn["label"].values
        Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, yd, test_size=0.2, random_state=42)

        # adapt k to dataset size
        k = min(3, len(Xd_train))
        clf_dynamic = KNeighborsClassifier(n_neighbors=k).fit(Xd_train, yd_train)
        print("Dynamic classifier trained, acc:", clf_dynamic.score(Xd_test, yd_test))

# === Setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create CSVs with header if not exists
if not os.path.exists(STATIC_FILE):
    with open(STATIC_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"] + [f"f{i}" for i in range(126)]
        writer.writerow(header)

if not os.path.exists(DYNAMIC_FILE):
    with open(DYNAMIC_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"] + [f"f{i}" for i in range(126*30)]
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

def draw_info(frame, msg=None):
    text = msg if msg else "Press 0-9 for static, G to record dynamic, Q to quit"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,255,255), 2, cv2.LINE_AA)
    return frame

# === Recording UI State ===
STATE = "idle"
countdown_start = None
recorded_frames = []
READY_DELAY = 3
RECORD_FRAMES = 30
border_color = None
dynamic_buffer = []

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

        # draw landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        feat_vec = features_from_result(result)

        # === Idle ===
        if STATE == "idle":
            frame = draw_info(frame)

            # Static prediction
            if clf_static is not None and np.any(feat_vec):
                pred_static = clf_static.predict([feat_vec])[0]
                cv2.putText(frame, f"Static: {pred_static}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)

            # Dynamic prediction (rolling buffer)
            if clf_dynamic is not None and np.any(feat_vec):
                dynamic_buffer.append(feat_vec)
                if len(dynamic_buffer) > RECORD_FRAMES:
                    dynamic_buffer.pop(0)

                if len(dynamic_buffer) == RECORD_FRAMES:
                    sample = np.concatenate(dynamic_buffer)
                    pred_dynamic = clf_dynamic.predict([sample])[0]
                    cv2.putText(frame, f"Dynamic: {pred_dynamic}", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 3, cv2.LINE_AA)

        # === Countdown ===
        elif STATE == "countdown":
            elapsed = time.time() - countdown_start
            remaining = READY_DELAY - int(elapsed)
            border_color = (0, 165, 255)  # orange
            cv2.putText(frame, f"Get Ready: {remaining}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, border_color, 3, cv2.LINE_AA)
            if elapsed >= READY_DELAY:
                STATE = "recording"
                recorded_frames = []

        # === Recording ===
        elif STATE == "recording":
            border_color = (0, 255, 0)  # green
            if np.any(feat_vec):
                recorded_frames.append(feat_vec)
            cv2.putText(frame, f"Recording... {len(recorded_frames)}/{RECORD_FRAMES}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, border_color, 3, cv2.LINE_AA)
            if len(recorded_frames) >= RECORD_FRAMES:
                STATE = "naming"

        # === Naming ===
        elif STATE == "naming":
            border_color = None
            cv2.putText(frame, "Press S to save, D to discard", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3, cv2.LINE_AA)

        # Draw border if needed
        if border_color is not None:
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0,0), (w-1,h-1), border_color, 10)

        cv2.imshow("Hand Tracking + Data Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # Save static gesture
        elif ord('0') <= key <= ord('9') and STATE == "idle":
            label = chr(key)
            if np.any(feat_vec):
                with open(STATIC_FILE, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    row = [label] + feat_vec.tolist()
                    writer.writerow(row)
                print(f"[SAVED] Static gesture {label}")

        # Start dynamic recording
        elif key == ord('g') and STATE == "idle":
            STATE = "countdown"
            countdown_start = time.time()

        # Save/Delete after recording
        elif STATE == "naming":
            if key == ord('s'):
                gesture_label = input("Enter dynamic gesture name: ")
                sample = np.concatenate(recorded_frames)
                with open(DYNAMIC_FILE, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    row = [gesture_label] + sample.tolist()
                    writer.writerow(row)
                print(f"[SAVED] Dynamic gesture {gesture_label}")
                STATE = "idle"
            elif key == ord('d'):
                print("[DISCARDED]")
                STATE = "idle"

cap.release()
cv2.destroyAllWindows()