import cv2
import time
import pyautogui

from hand_tracking import HandTracker


# --------------------------------------------------
# 1. Model path
# --------------------------------------------------

MODEL_PATH = "models/hand_landmarker.task"


# --------------------------------------------------
# 2. Create HandTracker
# --------------------------------------------------

tracker = HandTracker(MODEL_PATH)


# --------------------------------------------------
# 3. Get screen size
# --------------------------------------------------

screen_width, screen_height = pyautogui.size()

print(f"Screen Size: {screen_width} x {screen_height}")


# --------------------------------------------------
# 4. PyAutoGUI settings
# --------------------------------------------------

pyautogui.PAUSE = 0


# --------------------------------------------------
# 5. Smoothing settings
# --------------------------------------------------

smooth_factor = 0.25

previous_x = 0
previous_y = 0


# --------------------------------------------------
# 6. Camera area
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    tracker.close()
    exit()


# --------------------------------------------------
# 7. Main loop
# --------------------------------------------------

try:

    while True:

        # Read webcam frame
        success, frame = cap.read()

        if not success:
            print("Error: Could not read frame.")
            break


        # Mirror camera
        frame = cv2.flip(frame, 1)


        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # Timestamp
        timestamp_ms = int(time.monotonic() * 1000)


        # Detect hand
        result = tracker.detect(
            rgb_frame,
            timestamp_ms
        )


        # --------------------------------------------------
        # If hand detected
        # --------------------------------------------------

        if result.hand_landmarks:

            # Get first hand
            hand = result.hand_landmarks[0]


            # Landmark 8 = Index fingertip
            index_tip = hand[8]


            # Normalized coordinates
            x = index_tip.x
            y = index_tip.y


            # --------------------------------------------------
            # Camera pixel coordinates
            # --------------------------------------------------

            frame_height, frame_width, _ = frame.shape

            pixel_x = int(x * frame_width)
            pixel_y = int(y * frame_height)


            # --------------------------------------------------
            # Screen coordinates
            # --------------------------------------------------

            target_x = int(x * screen_width)
            target_y = int(y * screen_height)


            # --------------------------------------------------
            # Smooth cursor movement
            # --------------------------------------------------

            smooth_x = (
                previous_x
                + (target_x - previous_x) * smooth_factor
            )

            smooth_y = (
                previous_y
                + (target_y - previous_y) * smooth_factor
            )


            # Convert to integer
            smooth_x = int(smooth_x)
            smooth_y = int(smooth_y)


            # Save for next frame
            previous_x = smooth_x
            previous_y = smooth_y


            # --------------------------------------------------
            # Move actual mouse cursor
            # --------------------------------------------------

            pyautogui.moveTo(
                smooth_x,
                smooth_y
            )


            # --------------------------------------------------
            # Draw fingertip
            # --------------------------------------------------

            cv2.circle(
                frame,
                (pixel_x, pixel_y),
                10,
                (0, 255, 255),
                -1
            )


            # --------------------------------------------------
            # Display coordinates
            # --------------------------------------------------

            cv2.putText(
                frame,
                f"Index: X={x:.3f} Y={y:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Cursor: X={smooth_x} Y={smooth_y}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


        else:

            # No hand detected
            cv2.putText(
                frame,
                "No hand detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


        # --------------------------------------------------
        # Show camera
        # --------------------------------------------------

        cv2.imshow(
            "Remote Cursor Controller",
            frame
        )


        # --------------------------------------------------
        # Press Q to exit
        # --------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# --------------------------------------------------
# 8. Cleanup
# --------------------------------------------------

finally:

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()