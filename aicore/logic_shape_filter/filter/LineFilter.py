from typing import List, Dict, Set, Any

from aicore.component.DetectedObject import DetectedObject
from aicore.component.Line import Line
from aicore.component.Point import Point
from aicore.component.Trajectory import Trajectory
from aicore.logic_shape_filter.logic_shape_filter import LogicShapeFilter


class LineFilter(LogicShapeFilter):
    def __init__(self, name: str, id: int, location: Line, created_at: int, updated_at: int, attribute: List[str],
                 direction: int):
        super().__init__(name, id, location, created_at, updated_at)
        self.attr = attribute
        self.direction = direction

    def make_filtering(self, source_filter: Dict[Any, DetectedObject], current_frame = None):
        output = dict()
        for track_id, detectedObject in source_filter.items():
            trajectory_list = detectedObject.get_trajectories()
            output_set = set()

            if self._check_crossing_line(trajectory_list, output_set):
                iterator = iter(output_set)
                first_traj = next(iterator)
                second_traj = next(iterator)
                if self.direction == 0:
                    output[detectedObject.get_track_id()] = detectedObject
                elif self.direction == self._check_crossing_direction(first_traj, second_traj):
                    output[detectedObject.get_track_id()] = detectedObject
                else:
                    pass

        return output

    def _check_crossing_line(self, list_track: List[Trajectory], output: Set[Trajectory]):
        tmp = None
        for trajectory in list_track:
            if tmp is None:
                tmp = trajectory
                continue
            if self._is_line_cross(tmp.get_position(), trajectory.get_position()):
                output.add(tmp)
                output.add(trajectory)
                return True
            tmp = trajectory
        return False

    def _is_line_cross(self, point1: Point, point2: Point):
        line_start_point = self.location.get_start_point()
        line_end_point = self.location.get_end_point()

        check1 = (point1.Y - point2.Y) * (line_start_point.X - point1.X) - (line_start_point.Y - point1.Y) * (
                point1.X - point2.X)
        check2 = (point1.Y - point2.Y) * (line_end_point.X - point1.X) - (line_end_point.Y - point1.Y) * (
                point1.X - point2.X)
        check3 = (line_start_point.Y - line_end_point.Y) * (point1.X - line_start_point.X) - (
                point1.Y - line_start_point.Y) * (line_start_point.X - line_end_point.X)
        check4 = (line_start_point.Y - line_end_point.Y) * (point2.X - line_start_point.X) - (
                point2.Y - line_start_point.Y) * (line_start_point.X - line_end_point.X)

        return check1 * check2 <= 0 and check3 * check4 <= 0

    def _check_crossing_direction(self, first_trajectory: Trajectory, second_trajectory: Trajectory):
        line_start_point = self.location.get_start_point()
        orient = (first_trajectory.get_position().Y - second_trajectory.get_position().Y) * (
                    line_start_point.X - first_trajectory.get_position().X) - (
                             line_start_point.Y - first_trajectory.get_position().Y) * (
                             first_trajectory.get_position().X - second_trajectory.get_position().X)
        return 1 if orient >= 0 else -1
