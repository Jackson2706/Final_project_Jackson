from typing import Dict, Any, Union
from aicore.light_detection.light_detection import LightDetector
from aicore.component.DetectedObject import DetectedObject
from aicore.component.Line import Line
from aicore.component.Rect import Rect
from aicore.logic_shape_filter.logic_shape_filter import LogicShapeFilter


class LightFilter(LogicShapeFilter):
    def __init__(self, name: str, id: int, location: Union[Line, Rect], created_at: int, updated_at: int):
        super().__init__(name, id, location, created_at, updated_at)
        self.light_detector = LightDetector(location=location, threshold=0.01)

    def make_filtering(self, source_filter: Dict[Any, DetectedObject], current_frame=None):
        if self.light_detector.run_inference(current_frame):
            return source_filter
        else:
            return dict()

    def equals(self):
        super().equals()