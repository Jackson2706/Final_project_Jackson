from typing import Dict, Any, Union
from aicore.light_detection.light_detection import LightDetector
from aicore.component.DetectedObject import DetectedObject
from aicore.component.Line import Line
from aicore.component.Rect import Rect
from aicore.logic_shape_filter.logic_shape_filter import LogicShapeFilter

class LightFilter(LogicShapeFilter):
    def __init__(self, name: str, id: int, location: Union[Line, Rect], created_at: int, updated_at: int):
        """
        Initialize the LightFilter object.

        Parameters:
        - name (str): The name of the filter.
        - id (int): The unique identifier for the filter.
        - location (Union[Line, Rect]): The location associated with the filter.
        - created_at (int): The timestamp when the filter was created.
        - updated_at (int): The timestamp when the filter was last updated.
        """
        super().__init__(name, id, location, created_at, updated_at)
        self.light_detector = LightDetector(location=location, threshold=0.01)

    def make_filtering(self, source_filter: Dict[Any, DetectedObject], current_frame=None) -> Dict[Any, DetectedObject]:
        """
        Perform filtering based on light detection.

        Parameters:
        - source_filter (Dict[Any, DetectedObject]): The source filter containing detected objects.
        - current_frame: The current frame of image data for light detection inference.

        Returns:
        - Dict[Any, DetectedObject]: The filtered dictionary of detected objects. If light is detected, return the original
          source_filter. Otherwise, return an empty dictionary.
        """
        if self.light_detector.run_inference(current_frame):
            return source_filter
        else:
            return dict()

    def equals(self):
        super().equals()