from aicore.component.Point import Point
class Line:
    def __init__(self, start_point: Point, end_point: Point):
        self._start_point = start_point
        self._end_point = end_point

    def __str__(self):
        return f"Line: From {self._start_point} to {self._end_point}"

    def get_start_point(self):
        return self._start_point

    def get_end_point(self):
        return self._end_point
