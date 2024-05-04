from datetime import datetime

from aicore.component.trajectory import Trajectory


class DetectedObject(Trajectory):
    def __init__(self, track_id: int, x_center: int, y_center: int, width: int, height: int, label: str, conf: float,
                 time_stamp: datetime):
        super().__init__(x_center, y_center, width, height, label, conf, time_stamp)
        self.track_id = track_id

    def __str__(self):
        return f"Detected Object: [Track id: {self.track_id}, Label: {self.label}, Confidence: {self.conf}, Bounding Box: " \
               f"(X: {self.x_center}, Y: {self.y_center}, Width: {self.width}, Height: {self.height}), " \
               f"Timestamp: {self.timestamp}]"

    def get_infor(self):
        return self.track_id, self.x_center, self.y_center, self.width, self.height, self.label, self.conf, self.timestamp
