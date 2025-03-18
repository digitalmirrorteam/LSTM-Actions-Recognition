import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf
import threading
import h5py
import json

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

# Load the model directly
model = tf.keras.models.load_model("lstm-hand-punch.h5")
print(model.summary())

lm_list = []
label = "not grasped"
neutral_label = "not grasped"

def make_landmark_timestep(results):
    c_lm = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                c_lm.append(lm.x)
                c_lm.append(lm.y)
                c_lm.append(lm.z)
    else:
        # If no hands are detected, append zeros or a neutral value
        c_lm = [0.0] * 126  # Adjust this based on your model's input size
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    for hand_landmarks in results.multi_hand_landmarks:
        mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
    return frame

def draw_bounding_box_and_label(frame, results, label):
    for hand_landmarks in results.multi_hand_landmarks:
        x_min, y_min = 1, 1
        x_max, y_max = 0, 0
        for lm in hand_landmarks.landmark:
            x_min = min(x_min, lm.x)
            y_min = min(y_min, lm.y)
            x_max = max(x_max, lm.x)
            y_max = max(y_max, lm.y)
        h, w, c = frame.shape
        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)
        color = (0, 0, 255) if label != neutral_label else (0, 255, 0)
        thickness = 2 if label != neutral_label else 1
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.putText(frame, f"Status: {label}", (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    return frame

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    print(f"Input shape before expansion: {lm_list.shape}")  # Debug statement
    if lm_list.shape != (20, 126):  # Adjust this shape based on your model's input
        print(f"Invalid input shape: {lm_list.shape}")
        return neutral_label
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    percentage_result = result * 100
    print(f"Model prediction result: {percentage_result}")
    if result[0][0] > 0.5:
        label = "left_punch"
    elif result[0][1] > 0.5:
        label = "right_punch"
    return str(label)

i = 0
warm_up_frames = 60

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    i += 1
    if i > warm_up_frames:
        if results.multi_hand_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            if len(lm_list) == 20:
                t1 = threading.Thread(target=detect, args=(model, lm_list))
                t1.start()
                lm_list = []
            frame = draw_landmark_on_image(mpDraw, results, frame)
            frame = draw_bounding_box_and_label(frame, results, label)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()