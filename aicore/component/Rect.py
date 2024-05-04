from aicore.component.Point import Point
class Rect:
    def __init__(self, top_left_point: Point, bottom_right_point: Point):
        self.top_left_point = top_left_point
        self.bottom_right_point = bottom_right_point

    def __str__(self):
        return f"Rect: Top Left {self.top_left_point} - Bottom Right {self.bottom_right_point}"