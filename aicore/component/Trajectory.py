from datetime import datetime

from aicore.component.Point import Point


class Trajectory:
    def __init__(self, x_center: int, y_center: int, width: int, height: int, label: str, conf: float,
                 time_stamp: datetime):
        """
        Represents a trajectory with a bounding box, a label, and a timestamp.

        Parameters:
        - x_center (int): The x-coordinate of the center of the bounding box.
        - y_center (int): The y-coordinate of the center of the bounding box.
        - width (int): The width of the bounding box.
        - height (int): The height of the bounding box.
        - label (int): The label or class of the trajectory.
        - time_stamp (datetime): The timestamp indicating when the trajectory was observed.
        """
        self._x_center = x_center
        self._y_center = y_center
        self._width = width
        self._height = height
        self._label = label
        self._conf = conf
        self._timestamp = time_stamp

    def get_trajectory(self):
        return [self._x_center, self._y_center, self._width, self._height, self._label, self._conf, self._timestamp]

    def get_position(self):
        return Point(self._x_center, self._y_center)

    def get_label(self):
        return self._label

    def get_bounding_box(self):
        """
        Get the bounding box coordinates.

        Returns:
        - tuple: A tuple containing (x_center, y_center, width, height) of the bounding box.
        """
        return [self._x_center, self._y_center, self._width, self._height]

    def get_confidence_score(self):
        return self._conf

    def get_timestamp(self):
        """
        Get the timestamp of the trajectory.

        Returns:
        - datetime: The timestamp indicating when the trajectory was observed.
        """
        return self._timestamp

    def __str__(self):
        """
        Returns a string representation of the Trajectory object.
        """
        return f"Trajectory: [Label: {self._label}, Confidence: {self._conf}, Bounding Box: " \
               f"(X: {self._x_center}, Y: {self._y_center}, Width: {self._width}, Height: {self._height}), " \
               f"Timestamp: {self._timestamp}]"
