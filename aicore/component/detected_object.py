from typing import List
from math import sqrt
from aicore.component.trajectory import Trajectory

class Detected_Object:
    def __init__(self, track_id: int , label: str, trajectories: List[Trajectory]) -> None:
        """
        Represents a detected object with a unique track ID, label, and list of trajectories.

        Parameters:
        - track_id (int): The unique identifier for the detected object.
        - label (str): The label or class of the detected object.
        - trajectories (List[Trajectory]): A list of Trajectory objects representing the object's motion over time.
        """
        self.track_id = track_id
        self.trajectories = trajectories
        self.label = label
        self.statement = 0 # default 0: unknown      1: moving       2: stationary

        self.sensitive = 15
        self.stationary_threshold = 5

    def get_track_id(self):
        """
        Get the track ID of the detected object.

        Returns:
        - int: The unique identifier for the detected object.
        """
        return self.track_id
    
    def add_trajectory(self, trajectory):
        self.trajectories.append(trajectory)

    def get_trajectories(self):
        """
        Get the list of trajectories of the detected object.

        Returns:
        - List[Trajectory]: A list of Trajectory objects representing the object's motion over time.
        """
        return self.trajectories
    
    def get_current_speed(self):
        """
        Calculate the current speed of the detected object based on its trajectories.

        Returns:
        - float: The current speed of the detected object.
        """
        sum_distance = 0
        total_time = 0
        middle_index = len(self.trajectories)//2
        start = max(middle_index-self.sensitive, 0)
        end = min(middle_index+ self.sensitive - 1, len(self.trajectories) - 1)
        for i in range(start, end):
            current_trajectory = self.trajectories[i]
            x_current, y_current,_,_,_ = current_trajectory.get_bounding_box()
            next_trajectory = self.trajectories[i+1]
            x_next, y_next, _, _, _ = next_trajectory.get_bounding_box()
            sum_distance += sqrt((x_next-x_current)**2 + (y_next-y_current)**2)
            total_time += (next_trajectory.get_timestamp() - current_trajectory.get_timestamp())

        current_speed = sum_distance/total_time
        if current_speed > self.stationary_threshold:
            self.statement = 1
        else:
            self.statement = 2
        
        return current_speed
    
    def get_statement(self):
        """
        Get the statement (moving or stationary) of the detected object based on its current speed.

        Returns:
        - int: The statement of the detected object.
          - 0: Unknown
          - 1: Moving
          - 2: Stationary
        """
        _ = self.get_current_speed()
        return self.statement

    def get_last_trajectory(self):
        """
        Get the timestamp of the last Trajectory in the list.

        Returns:
        - datetime: The timestamp of the last Trajectory.
        """
        if not self.trajectories:
            return None
        return self.trajectories[-1]
    
    def __str__(self):
        """
        String representation of the Detected_Object.

        Returns:
        - str: A string containing information about the Detected_Object.
        """
        trajectory_info = "\n".join([f"Trajectory {i+1}: {trajectory}" for i, trajectory in enumerate(self.trajectories)])
        return f"Detected_Object - Track ID: {self.track_id}, Label: {self.label}, Statement: {self.statement}\nTrajectories:\n{trajectory_info}"