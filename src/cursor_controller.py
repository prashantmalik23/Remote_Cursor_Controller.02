import math
import pyautogui


class CursorController:

    # --- Tunable settings ---

    # Smoothing used during normal (non-drag) cursor movement.
    # Lower = smoother/less jittery but slower to catch up.
    NORMAL_SMOOTHING = 0.2

    # Smoothing used while dragging AND the hand is moving fast.
    DRAG_SMOOTHING = 0.6

    # Smoothing used while dragging AND the hand is moving slowly.
    MIN_DRAG_SMOOTHING = 0.35

    # Minimum pixel distance the cursor must move before we actually
    # send a moveTo() call.
    MOVEMENT_THRESHOLD = 4

    # Speed at or above which dragging uses full DRAG_SMOOTHING.
    SPEED_THRESHOLD = 20

    # --- Scroll settings ---

    # Controls how strongly each scroll command moves the page.
    # Higher = faster scrolling.
    SCROLL_SENSITIVITY = 1

    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()

        # Tracks the last actual cursor position.
        self.prev_screen_x = None
        self.prev_screen_y = None

        # Automatically tracked drag state.
        self.is_dragging = False

    def _get_drag_alpha(self, speed):
        """
        Turns the current movement speed into a smoothing value
        between MIN_DRAG_SMOOTHING and DRAG_SMOOTHING.
        """

        speed_ratio = min(speed / self.SPEED_THRESHOLD, 1.0)

        alpha = (
            self.MIN_DRAG_SMOOTHING
            + speed_ratio
            * (self.DRAG_SMOOTHING - self.MIN_DRAG_SMOOTHING)
        )

        return alpha

    def move_cursor(self, x, y, frame_width, frame_height):
        # Step 1: Convert fingertip coordinates -> screen coordinates
        raw_x = int(x / frame_width * self.screen_width)
        raw_y = int(y / frame_height * self.screen_height)

        # First frame
        if self.prev_screen_x is None or self.prev_screen_y is None:
            self.prev_screen_x = raw_x
            self.prev_screen_y = raw_y

            pyautogui.moveTo(raw_x, raw_y)
            return

        # Step 2: Measure movement speed
        speed = math.hypot(
            raw_x - self.prev_screen_x,
            raw_y - self.prev_screen_y
        )

        # Step 3: Select smoothing
        if self.is_dragging:
            alpha = self._get_drag_alpha(speed)
        else:
            alpha = self.NORMAL_SMOOTHING

        # Step 4: Exponential smoothing
        smoothed_x = int(
            self.prev_screen_x
            + alpha * (raw_x - self.prev_screen_x)
        )

        smoothed_y = int(
            self.prev_screen_y
            + alpha * (raw_y - self.prev_screen_y)
        )

        # Step 5: Movement threshold
        move_distance = math.hypot(
            smoothed_x - self.prev_screen_x,
            smoothed_y - self.prev_screen_y
        )

        if move_distance < self.MOVEMENT_THRESHOLD:
            return

        pyautogui.moveTo(smoothed_x, smoothed_y)

        self.prev_screen_x = smoothed_x
        self.prev_screen_y = smoothed_y

    # --------------------------------------------------
    # Click functions
    # --------------------------------------------------

    def left_click(self):
        pyautogui.click()

    def right_click(self):
        pyautogui.rightClick()

    # --------------------------------------------------
    # Drag and Drop
    # --------------------------------------------------

    def mouse_down(self):
        self.is_dragging = True
        pyautogui.mouseDown()

    def mouse_up(self):
        self.is_dragging = False
        pyautogui.mouseUp()

    # --------------------------------------------------
    # Scroll
    # --------------------------------------------------

    def scroll(self, amount):
        """
        Scroll the mouse wheel.

        Positive amount  -> Scroll UP
        Negative amount  -> Scroll DOWN
        """

        scroll_amount = int(amount * self.SCROLL_SENSITIVITY)

        if scroll_amount != 0:
            pyautogui.scroll(scroll_amount)