import cv2
import mediapipe as mp
import time


MODEL_PATH = "models/hand_landmarker.task"


def draw_hand_landmarks(frame, landmarks):
    """Draw the 21 hand landmarks and their connections."""

    height, width, _ = fr