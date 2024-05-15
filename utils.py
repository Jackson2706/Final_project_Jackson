from threading import Thread
import cv2
from typing import Dict
from application.streaming.streaming import IPCameraFrameRecoder
from aicore.component.DetectedObject import DetectedObject

class ViolationVideoSaver(Thread):
    def __init__(self, save_path: str, cam_object: IPCameraFrameRecoder, selected_obj: DetectedObject, filter_dict: Dict):
        super().__init__()
        self.save_path = save_path
        self.cam_obj = cam_object
        self.selected_obj = selected_obj
        self.running = True
        self.filter_dict = filter_dict

    @staticmethod
    def draw_box(frame, bbox):
        # Function to draw bounding box on the frame
        x_center, y_center, w, h = bbox
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color, thickness 2
        return frame

    def run(self):
        out = None
        # Get trajectory list and start/end time
        trajectort_list = self.selected_obj.get_trajectories()
        start_time = trajectort_list[0].get_timestamp()
        end_time = trajectort_list[-1].get_timestamp()
        # Extract frames between start and end time
        frame_queue = self.cam_obj.extract_frames_between_timestamps(start_time, end_time)

        # Determine the total number of frames required
        total_frames = len(frame_queue)

        # Run the loop while there are frames in the queue or trajectories in the list
        while self.running and (total_frames > 0 or trajectort_list):
            if frame_queue:
                frame = frame_queue.pop(0)
                # Create video writer object
                if out is None:
                    h, w, _ = frame.shape
                    out = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (w, h))

                # Draw bounding box if trajectory list is not empty
                if trajectort_list:
                    bbox = trajectort_list.pop(0).get_bounding_box()
                    frame = self.draw_box(frame, bbox)

                # Draw filters on the frame
                for id, filter in self.filter_dict.items():
                    frame = filter.draw(frame)

                # Write frame to video
                out.write(frame)
                total_frames -= 1

        # Release video writer object when done
        if out:
            out.release()

    def stop(self):
        # Method to stop the thread
        self.running = False
        self.join()
