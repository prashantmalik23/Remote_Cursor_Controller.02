import mediapipe as mp


class HandTracker:

    def __init__(self, model_path):
        # MediaPipe classes
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Configure the hand detector
        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Create hand detector
        self.landmarker = HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame, timestamp_ms):
        """
        Detect hand landmarks from an RGB frame.
        """

        # Convert OpenCV image to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand landmarks
        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        return result

    def close(self):
        """Release MediaPipe resources."""

        self.landmarker.close()