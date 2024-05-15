from typing import List
from math import hypot
from aicore.component.Trajectory import Trajectory
from aicore.component.DetectedObject import DetectedObject


class KalmanFilterTracking:
    def __init__(self, distance_thresh: int):
        """
        Initialize KalmanFilterTracking object.

        Parameters:
        - distance_thresh (int): Threshold distance to determine if two trajectories correspond to the same object.
        """
        self.distance_threshold = distance_thresh
        self.count = 0
        self.tracking_object = {}
        self.track_id = 0
        self.previous_detection_result = []

    def update(self, bboxes_traject: List[Trajectory], image=None):
        """
        Update the tracking based on the new detections.

        Parameters:
        - bboxes_traject (List[Trajectory]): List of Trajectory objects representing the detections in the current frame.
        - image: Optional parameter for additional image data (not used in this method).

        Returns:
        - Dict[int, DetectedObject]: Dictionary containing the updated tracking information.
        """
        self.count += 1

        # At the beginning or when there are only 1 or 2 frames, compare previous and current detections.
        if self.count <= 2:
            for obj in bboxes_traject:
                pt = obj.get_position()
                for obj2 in self.previous_detection_result:
                    pt2 = obj2.get_position()
                    distance = hypot(pt2.get_x() - pt.get_x(), pt2.get_y() - pt.get_y())
                    # If the distance is below the threshold, update object position.
                    if distance < self.distance_threshold:
                        self.tracking_object[self.track_id] = DetectedObject(track_id=self.track_id,
                                                                             label=obj.get_label(), trajectories=[obj])
                        self.track_id += 1
        else:
            # Make copies to avoid modifying the original lists.
            tracking_object_copy = self.tracking_object.copy()
            bboxes_traject_copy = bboxes_traject.copy()
            for track_id, obj2 in tracking_object_copy.items():
                if len(obj2.get_trajectories()) == 0:
                    continue
                pt2 = obj2.get_last_trajectory().get_position()
                object_exists = False
                for obj in bboxes_traject_copy:
                    pt = obj.get_position()
                    distance = hypot(pt2.get_x() - pt.get_x(), pt2.get_y() - pt.get_y())
                    # If the distance is below the threshold, add trajectory to the existing object.
                    if distance < self.distance_threshold:
                        self.tracking_object[track_id].add_trajectory(obj)
                        object_exists = True
                        # Remove the detected object from the current frame's detections.
                        if obj in bboxes_traject:
                            bboxes_traject.remove(obj)
                        continue
                # If the object is not found in the current frame's detections, remove it from the tracking.
                if not object_exists:
                    self.tracking_object.pop(track_id)

        # Add new IDs for new detections.
        for obj in bboxes_traject:
            self.tracking_object[self.track_id] = DetectedObject(track_id=self.track_id, label=obj.get_label(),
                                                                 trajectories=[obj])
            self.track_id += 1
        # Update previous detections to be used in the next iteration.
        self.previous_detection_result = bboxes_traject.copy()
        return self.tracking_object
