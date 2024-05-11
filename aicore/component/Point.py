class Point:
    def __init__(self, X, Y):
        self._X = X
        self._Y = Y

    def get_point(self):
        return self._X, self._Y

    def __str__(self):
        return f"Point: ({self._X}, {self._Y}"
