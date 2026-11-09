import cv2
import mediapipe as mp
import time
from .light import LightController

# Gesture mapping:
#   Open palm (4 extended fingers) -> LIGHT ON
#   Closed fist (0 extended fingers) -> LIGHT OFF
#   Other hand poses -> keep previous state

class HandGestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.60,
        )
        self.drawer = mp.solutions.drawing_utils
        self.light = LightController()
        self.last_gesture = "Show an open palm or fist"
        self.last_change = 0.0

    @staticmethod
    def _finger_states(hand_landmarks):
        lm = hand_landmarks.landmark
        # Index, middle, ring and pinky: tip above the PIP joint means extended.
        states = []
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            states.append(lm[tip].y < lm[pip].y)
        # Thumb is omitted because its orientation changes with handedness/camera.
        return states

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            self.drawer.draw_landmarks(
                frame, hand, self.mp_hands.HAND_CONNECTIONS
            )
            fingers = self._finger_states(hand)
            extended = sum(fingers)

            # Require a stable pose briefly to reduce accidental toggles.
            now = time.time()
            if now - self.last_change > 0.7:
                if extended == 4 and not self.light.is_on:
                    self.light.turn_on()
                    self.last_gesture = "OPEN PALM - LIGHT ON"
                    self.last_change = now
                elif extended == 0 and self.light.is_on:
                    self.light.turn_off()
                    self.last_gesture = "FIST - LIGHT OFF"
                    self.last_change = now
                elif extended == 4:
                    self.last_gesture = "OPEN PALM - LIGHT ON"
                elif extended == 0:
                    self.last_gesture = "FIST - LIGHT OFF"
                else:
                    self.last_gesture = f"Hand detected ({extended}/4 fingers)"
        return frame

    def close(self):
        self.hands.close()


def run(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions or change camera_index."
        )

    controller = HandGestureController()
    print("Hand Gesture Control started. Press Q or ESC to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Unable to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            frame = controller.process(frame)

            status = "ON" if controller.light.is_on else "OFF"
            color = (70, 220, 100) if controller.light.is_on else (60, 90, 240)
            cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, 100), (20, 35, 55), -1)
            cv2.putText(frame, "HAND GESTURE CONTROL", (25, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 240, 130), 2)
            cv2.putText(frame, f"LIGHT: {status}", (25, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, controller.last_gesture, (25, frame.shape[0]-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

            cv2.imshow("Hand Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        controller.close()
        cap.release()
        cv2.destroyAllWindows()
