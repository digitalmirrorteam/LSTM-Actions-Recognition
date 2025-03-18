import cv2
import mediapipe as mp
import pandas as pd

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) 
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "h_neutral"
no_of_frames = 500

def make_landmark_timestep(results):
    c_lm = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                c_lm.append(lm.x)
                c_lm.append(lm.y)
                c_lm.append(lm.z)
    return c_lm

def draw_landmark_on_image(mpDraw, results, frame):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    if ret:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)
        if results.multi_hand_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            frame = draw_landmark_on_image(mpDraw, results, frame)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()
