from datetime import datetime

class Trajectory:
    def __init__(self, x_center: int, y_center: int, width: int, height: int, label: int, time_stamp: datetime):
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
        self.timestamp = time_stamp

    def get_bounding_box(self):
        """
        Get the bounding box coordinates and label.

        Returns:
        - tuple: A tuple containing (x_center, y_center, width, height, label) of the bounding box.
        """
        return (self.x_center, self.y_center, self.width, self.height, self.label)
    
    def get_timestamp(self):
        """
        Get the timestamp of the trajectory.

        Returns:
        - datetime: The timestamp indicating when the trajectory was observed.
        """
        return self.timestamp
