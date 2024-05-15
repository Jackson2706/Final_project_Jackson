from abc import ABC
from typing import Union, Dict, Any
import cv2
from aicore.component.DetectedObject import DetectedObject
from aicore.component.Line import Line
from aicore.component.Rect import Rect


class LogicShapeFilter(ABC):
    def __init__(self, name: str, id: int, location: Union[Line, Rect], created_at: int, updated_at: int):
        self.name = name
        self.id = id
        self.location = location
        self.created = created_at
        self.updated = updated_at

    def make_filtering(self, source_filter: Dict[Any, DetectedObject], current_frame=None):
        pass

    def equals(self):
        pass

    def draw(self, image):
        """
        Draw the location on the image.

        Parameters:
        - image: The input image on which the location will be drawn.

        Returns:
        - The image with the location drawn on it.
        """
        if isinstance(self.location, Rect):
            # Draw rectangle
            top_left = (self.location.get_top_left_point().get_x(), self.location.get_top_left_point().get_y())
            bottom_right = (self.location.get_bottom_right_point().get_x(), self.location.get_bottom_right_point().get_y())
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(image, f'LightFiter: {self.id}', top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif isinstance(self.location, Line):
            # Draw line
            start_point = (self.location.get_start_point().get_x(), self.location.get_start_point().get_y())
            end_point = (self.location.get_end_point().get_x(), self.location.get_end_point().get_y())
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)
            cv2.putText(image, f'Line: {self.id}', start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image
