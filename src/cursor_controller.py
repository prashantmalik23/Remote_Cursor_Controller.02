import pyautogui


class CursorController:

    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()

    def move_cursor(self, x, y, frame_width, frame_height):

        screen_x = int(x / frame_width * self.screen_width)
        screen_y = int(y / frame_height * self.screen_height)

        pyautogui.moveTo(screen_x, screen_y)

    def left_click(self):
        pyautogui.click()

    def right_click(self):
        pyautogui.rightClick()

    def mouse_down(self):
        pyautogui.mouseDown()

    def mouse_up(self):
        pyautogui.mouseUp()    