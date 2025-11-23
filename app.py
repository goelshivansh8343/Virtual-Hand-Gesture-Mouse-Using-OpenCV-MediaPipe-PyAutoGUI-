import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # FIXED: Convert BGR → RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output = hands.process(rgb_frame)
    hand_landmarks = output.multi_hand_landmarks

    if hand_landmarks:
        for hand in hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            index_x, thumb_x, right_x = 0, 0, 0
            index_y, thumb_y, right_y = 0, 0, 0

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                # INDEX Finger
                if id == 8:
                    cv2.circle(frame, (x, y), 15, (255, 255, 255), -1)
                    index_x = int(screen_width / frame_width * x)
                    index_y = int(screen_height / frame_height * y)
                    pyautogui.moveTo(index_x, index_y)

                # THUMB Finger (Click detection)
                if id == 4:
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                    thumb_x = int(screen_width / frame_width * x)
                    thumb_y = int(screen_height / frame_height * y)
                    if abs(index_y - thumb_y) < 50:
                        pyautogui.click()
                        pyautogui.sleep(1)

                # PINKY Finger (Right-Click)
                if id == 20:
                    cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                    right_y = int(screen_height / frame_height * y)
                    if abs(index_y - right_y) < 50:
                        pyautogui.rightClick()
                        pyautogui.sleep(1)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
