import cv2
import mediapipe as mp
from HandTrackingModule import FindHands

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.75,static_image_mode=True)
detector = FindHands()
while True:
    ret, image = cap.read()
    hand = detector.getPosition(image, range(21), draw=False)
    # print("Index finger up:", detector.index_finger_up(image))
    # print("Middle finger up:", detector.middle_finger_up(image))

    if detector.index_finger_up(image) == True and detector.middle_finger_up(image) == True:
        print('2 jari')

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    cv2.imshow('Handtrackers', image)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp

# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands

# # Define function to count extended fingers
# def count_fingers(hand_landmarks):
#     # Tip and MCP landmarks for each finger
#     tips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
#             hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
#             hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
#             hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]
    
#     mcps = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
#             hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
#             hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
#             hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]]

#     count = 0
#     for tip, mcp in zip(tips, mcps):
#         if tip.y < mcp.y:  # Assuming the hand is upright
#             count += 1

#     # Thumb: special case
#     thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#     thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

#     if thumb_tip.x < thumb_ip.x if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < thumb_ip.x else thumb_tip.x > thumb_ip.x:  # Thumb direction differs
#         count += 1

#     return count

# cap = cv2.VideoCapture(0)
# hands = mp_hands.Hands()

# while True:
#     ret, image = cap.read()
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     result = hands.process(image)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS
#             )
#             finger_count = count_fingers(hand_landmarks)
#             cv2.putText(image, f'Fingers: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
#     cv2.imshow('handtrack', image)
#     if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
#         break

# cap.release()
# cv2.destroyAllWindows()
