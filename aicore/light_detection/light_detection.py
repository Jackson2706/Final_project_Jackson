import cv2
import numpy as np
from aicore.component.Rect import Rect  # Assuming this is a custom module


class LightDetector:
    def __init__(self, location: Rect, threshold: float):
        """
        Initialize the LightDetector object.

        Args:
            location (Rect): The location of the region of interest (ROI) where the detection will be performed.
            threshold (float): The threshold for determining whether the detected light is significant.
        """
        self._location = location
        self._threshold = threshold
        self._desired_dim = (30, 90)  # Desired dimensions for resizing the input image

    def run_inference(self, frame):
        """
        Perform light detection on the given frame.

        Args:
            frame (numpy.ndarray): The input frame to perform detection on.

        Returns:
            bool: True if significant light is detected, False otherwise.
        """
        # Extract ROI from the frame based on the specified location
        x_min, y_min = self._location.get_top_left_point().get_point()
        x_max, y_max = self._location.get_bottom_right_point().get_point()
        raw_image = frame[y_min:y_max, x_min:x_max]

        # Resize the image to desired dimensions
        img = cv2.resize(raw_image, self._desired_dim, interpolation=cv2.INTER_LINEAR)

        # Convert the image to HSV color space
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Define masks for detecting red and yellow colors
        # Lower mask for red (0-10)
        lower_red = np.array([0, 70, 50])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # Upper mask for red (170-180)
        lower_red1 = np.array([170, 70, 50])
        upper_red1 = np.array([180, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)

        # Mask for yellow color
        lower_yellow = np.array([21, 39, 64])
        upper_yellow = np.array([40, 255, 255])
        mask2 = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

        # Combine masks to detect red and yellow pixels
        mask = mask0 + mask1 + mask2

        # Calculate the percentage of red values in the masked region
        rate = np.count_nonzero(mask) / (self._desired_dim[0] * self._desired_dim[1])

        # Check if the percentage exceeds the threshold
        if rate > self._threshold:
            return True
        else:
            return False
