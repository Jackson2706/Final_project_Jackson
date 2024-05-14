from typing import List, Dict, Set, Any
from aicore.component.DetectedObject import DetectedObject
from aicore.component.Line import Line
from aicore.component.Point import Point
from aicore.component.Trajectory import Trajectory
from aicore.logic_shape_filter.logic_shape_filter import LogicShapeFilter


class LineFilter(LogicShapeFilter):
    def __init__(self, name: str, id: int, location: Line, created_at: int, updated_at: int, attribute: List[str],
                 direction: int):
        """
        Initialize the LineFilter object.

        Args:
            name (str): The name of the filter.
            id (int): The ID of the filter.
            location (Line): The line object representing the filtering area.
            created_at (int): The timestamp when the filter was created.
            updated_at (int): The timestamp when the filter was last updated.
            attribute (List[str]): The attributes associated with the filter.
            direction (int): The direction of the crossing line.
        """
        super().__init__(name, id, location, created_at, updated_at)
        self.attr = attribute
        self.direction = direction

    def make_filtering(self, source_filter: Dict[Any, DetectedObject], current_frame=None) -> Dict[Any, DetectedObject]:
        """
        Perform filtering based on line crossing.

        Args:
            source_filter (Dict[Any, DetectedObject]): A dictionary containing DetectedObject instances.
            current_frame: Optional argument for the current frame.

        Returns:
            Dict[Any, DetectedObject]: A dictionary containing filtered DetectedObject instances.
        """
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

    def _check_crossing_line(self, list_track: List[Trajectory], output: Set[Trajectory]) -> bool:
        """
        Check if any trajectory crosses the line.

        Args:
            list_track (List[Trajectory]): List of trajectory objects.
            output (Set[Trajectory]): Set to store trajectories that cross the line.

        Returns:
            bool: True if any trajectory crosses the line, False otherwise.
        """
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

    def _is_line_cross(self, point1: Point, point2: Point) -> bool:
        """
        Check if the line defined by two points crosses the filter line.

        Args:
            point1 (Point): First point.
            point2 (Point): Second point.

        Returns:
            bool: True if the line crosses the filter line, False otherwise.
        """
        line_start_point = self.location.get_start_point()
        line_end_point = self.location.get_end_point()

        check1 = (point1.get_y() - point2.get_y()) * (line_start_point.get_x() - point1.get_x()) - (line_start_point.get_y() - point1.get_y()) * (
                point1.get_x() - point2.get_x())
        check2 = (point1.get_y() - point2.get_y()) * (line_end_point.get_x() - point1.get_x()) - (line_end_point.get_y() - point1.get_y()) * (
                point1.get_x() - point2.get_x())
        check3 = (line_start_point.get_y() - line_end_point.get_y()) * (point1.get_x() - line_start_point.get_x()) - (
                point1.get_y() - line_start_point.get_y()) * (line_start_point.get_x() - line_end_point.get_x())
        check4 = (line_start_point.get_y() - line_end_point.get_y()) * (point2.get_x() - line_start_point.get_x()) - (
                point2.get_y() - line_start_point.get_y()) * (line_start_point.get_x() - line_end_point.get_x())

        return check1 * check2 <= 0 and check3 * check4 <= 0

    def _check_crossing_direction(self, first_trajectory: Trajectory, second_trajectory: Trajectory) -> int:
        """
        Check the direction of line crossing.

        Args:
            first_trajectory (Trajectory): First trajectory object.
            second_trajectory (Trajectory): Second trajectory object.

        Returns:
            int: 1 if the orientation is positive, -1 otherwise.
        """
        line_start_point = self.location.get_start_point()
        orient = (first_trajectory.get_position().get_y()- second_trajectory.get_position().get_y()) * (
                    line_start_point.get_x() - first_trajectory.get_position().get_x()) - (
                             line_start_point.get_y() - first_trajectory.get_position().get_y()) * (
                             first_trajectory.get_position().get_x() - second_trajectory.get_position().get_x())
        return 1 if orient >= 0 else -1
