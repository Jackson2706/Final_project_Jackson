from datetime import datetime

class Trajectory:
    def __init__(self, x_center: int, y_center: int, width: int, height: int, label: str, conf: float, time_stamp: datetime):
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
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.label = label
        self.conf = conf
        self.timestamp = time_stamp


    def get_trajectory(self):
        return [self.x_center, self.y_center, self.width, self.height, self.label, self.conf, self.timestamp]

    def get_bounding_box(self):
        """
        Get the bounding box coordinates.

        Returns:
        - tuple: A tuple containing (x_center, y_center, width, height) of the bounding box.
        """
        return [self.x_center, self.y_center, self.width, self.height]


    def get_confidence_score(self):
        return self.conf
    
    def get_timestamp(self):
        """
        Get the timestamp of the trajectory.

        Returns:
        - datetime: The timestamp indicating when the trajectory was observed.
        """
        return self.timestamp
    def __str__(self):
        """
        Returns a string representation of the Trajectory object.
        """
        return f"Trajectory: [Label: {self.label}, Confidence: {self.conf}, Bounding Box: " \
               f"(X: {self.x_center}, Y: {self.y_center}, Width: {self.width}, Height: {self.height}), " \
               f"Timestamp: {self.timestamp}]"