import cv2
import mediapipe as mp
import pygame
import numpy as np

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Pygame for sound
pygame.mixer.init()
sound_files = ["sa.wav", "re.wav", "ga.wav", "ma.wav", "pa.wav", "dha.wav", "ni.wav"]
sounds = [pygame.mixer.Sound(f"sounds/{file}") for file in sound_files]  # Load sounds

# Define 7 equal partitions (Adjust based on paper size)
WIDTH = 640  # Adjust based on your camera frame size
KEY_WIDTH = WIDTH // 7  # Dividing into 7 equal sections
KEYS = [(i * KEY_WIDTH, 300, (i + 1) * KEY_WIDTH, 400) for i in range(7)]

# Track last active key and current playing sound
last_key = None
current_sound = None

# Capture webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    current_key = None  # Reset current key each frame

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingertip = hand_landmarks.landmark[8]  # Index finger tip
            h, w, _ = frame.shape
            finger_x, finger_y = int(fingertip.x * w), int(fingertip.y * h)

            # Check if fingertip is on any partition
            for i, (x1, y1, x2, y2) in enumerate(KEYS):
                if x1 < finger_x < x2 and y1 < finger_y < y2:
                    current_key = i
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Highlight active key
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Draw keys

    # Sound logic: Play when hovering, stop when moving out
    if current_key is not None and current_key != last_key:  # New key detected
        if current_sound:
            current_sound.stop()  # Stop previous sound
        current_sound = sounds[current_key]  # Assign new sound
        current_sound.play(-1)  # Loop sound while hovering

    elif current_key is None and current_sound:  # Hand moved away
        current_sound.stop()
        current_sound = None

    last_key = current_key  # Update last key played

    # Display the frame
    cv2.imshow("Paper Flute Piano", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
