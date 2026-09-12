import cv2
import time

from hand_tracking import HandTracker


MODEL_PATH = "models/hand_landmarker.task"


def draw_hand_landmarks(frame, landmarks):
    """
    Draw the 21 hand landmarks.
    """

    height, width, _ = frame.shape

    points = []

    # Draw landmark points
    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

    # Connections between hand landmarks
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    # Draw landmark connections
    for start, end in connections:

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            2
        )


def main():

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    # Create HandTracker object
    tracker = HandTracker(MODEL_PATH)

    print("Hand tracking started.")
    print("Press 'q' to close.")

    while True:

        # Read webcam frame
        success, frame = cap.read()

        if not success:
            print("Error: Cannot receive frame.")
            break

        # Flip webcam for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create timestamp
        timestamp_ms = int(time.time() * 1000)

        # Detect hand
        result = tracker.detect(
            rgb_frame,
            timestamp_ms
        )

        # Check whether hand exists
        if result.hand_landmarks:

            # Loop through detected hands
            for hand_landmarks in result.hand_landmarks:

                # Draw landmarks
                draw_hand_landmarks(
                    frame,
                    hand_landmarks
                )

        # Display webcam
        cv2.imshow(
            "Remote Cursor Controller",
            frame
        )

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()