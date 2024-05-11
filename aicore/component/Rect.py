from aicore.component.Point import Point


class Rect:
    def __init__(self, top_left_point: Point, bottom_right_point: Point):
        self._top_left_point = top_left_point
        self._bottom_right_point = bottom_right_point

    def get_top_left_point(self):
        return self._top_left_point

    def get_bottom_right_point(self):
        return self._bottom_right_point

    def __str__(self):
        return f"Rect: Top Left {self._top_left_point} - Bottom Right {self._bottom_right_point}"
