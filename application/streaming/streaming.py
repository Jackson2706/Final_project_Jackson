from cv2 import VideoCapture, imwrite, imread
from collections import deque
from os.path import exists, join
from os import mkdir, remove
from time import time

from logger.system_logger import SystemLogger
class IPCameraFrameRecoder:
    def __init__(self,camera_url, max_queue_size, frame_dir, log_file):
        self.cap = VideoCapture(camera_url)
        self.max_queue_size = max_queue_size
        self.frame_dir = frame_dir
        self.frame_queue = deque(maxlen=self.max_queue_size)
        self.create_frame_directory()
        self.logger = SystemLogger(log_file=log_file)

    def create_frame_directory(self):
        '''
            This function is used to create folder saving frame from queue
        '''
        if not exists(self.frame_dir):
            mkdir(self.frame_dir)

    def save_frame(self, frame):
        '''
            This function is used to save the information about image frame to folder
            @param frame: current frame
        '''

        # Check if frame queue is full
        # if it's full, remove the oldest frame
        start_time = time()
        if(len(self.frame_queue) >= self.max_queue_size):
            removed_frame = self.frame_queue.popleft()
            remove(join(self.frame_dir, removed_frame["filename"]))
        

        # Case frame queue is not full
        # get timestamp (in miliseconds), store information about this frame to queue and save frame to folder
        current_time = time() * 1000
        file_name = f"frame_{current_time}.jpg"
        file_path = join(self.frame_dir, file_name)
        imwrite(file_path, frame)
        self.frame_queue.append(
            {
                "filename": file_name,
                "timestamp": current_time
            }
        )
        end_time = time()
        self.logger.log_info(f"Successfully save frame to {file_name} in {end_time - start_time} s")


    def read_frame(self):
        '''
            This function is used to get frame from cap
            It will create a generator and use key "yield" to get frame
        '''  
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.logger.log_error("Can not read frame from camera")
                break
            self.save_frame(frame=frame)

            yield frame
        

    def extract_frames_between_timestamps(self, beginTime, endTime):
        '''
            This function is used to extract frame in a block of time. to help make violation video
            @param: 
                beginTime: (in miliseconds): the timestamp starting violation
                endTime  : (in miliseconds): the timestamp ending violation 

            @return:
                a list of frame in range of time
        '''
        extracted_frames = []
        for frame_info in self.frame_queue:
            if beginTime <= frame_info["timestamp"] <= endTime:
                img = imread(join(self.FRAME_DIR, frame_info["filename"]))
                extracted_frames.append(img)
        return extracted_frames
    
    def release(self):
        self.cap.release()