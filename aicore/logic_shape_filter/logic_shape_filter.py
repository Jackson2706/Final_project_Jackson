from abc import ABC
from typing import Union, Dict, Any

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

    def make_filtering(self, source_filter: Dict[Any, DetectedObject], current_frame = None):
        pass

    def equals(self):
        pass