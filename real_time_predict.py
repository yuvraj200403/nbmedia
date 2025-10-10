import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)   # Speed of speech
engine.setProperty('volume', 1.0) # Max volume

# Load the trained model
model = joblib.load("gesture_model.pkl")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Gesture labels
labels = {
    0: "Hello",
    1: "Thanks",
    2: "I Love You"
}

# Start webcam
cap = cv2.VideoCapture(0)

# Voice output control variables
last_prediction = None
stable_count = 0
speak_threshold = 15  # Frames before first speech
last_spoken_time = 0
speak_interval = 0.5  # seconds between speaking same gesture again

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract 21 landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Predict gesture
            data = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(data)[0]

            # Stability check
            if prediction == last_prediction:
                stable_count += 1
            else:
                stable_count = 0
                last_prediction = prediction
                last_spoken_time = 0  # Reset so it speaks immediately for new gesture

            # Speak logic
            if stable_count >= speak_threshold:
                current_time = time.time()
                if current_time - last_spoken_time >= speak_interval:
                    text = labels.get(prediction, "Unknown")
                    print("Predicted:", text)
                    engine.say(text)
                    engine.runAndWait()
                    last_spoken_time = current_time  # update after speaking

            # Display gesture label
            label_text = labels.get(prediction, "Unknown")
            cv2.putText(frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
