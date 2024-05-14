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

        # Resize the image to desired dimensions directly in cv2.resize function
        img = cv2.resize(raw_image, self._desired_dim)

        # Convert the image to HSV color space
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Define ranges for red and yellow colors
        red_lower_range = np.array([[0, 70, 50], [10, 255, 255]])  # Lower range for red (0-10)
        red_upper_range = np.array([[170, 70, 50], [180, 255, 255]])  # Upper range for red (170-180)
        yellow_range = np.array([[21, 39, 64], [40, 255, 255]])  # Range for yellow color

        # Generate masks for red and yellow colors
        mask_red = cv2.inRange(img_hsv, red_lower_range[0], red_lower_range[1])  # Mask for lower red range
        mask_red += cv2.inRange(img_hsv, red_upper_range[0], red_upper_range[1])  # Mask for upper red range
        mask_yellow = cv2.inRange(img_hsv, yellow_range[0], yellow_range[1])  # Mask for yellow color

        # Combine masks to detect red and yellow pixels
        mask = mask_red + mask_yellow

        # Calculate the percentage of red values in the masked region
        red_pixel_count = np.count_nonzero(mask)

        # Calculate the total number of pixels in the region
        total_pixels = self._desired_dim[0] * self._desired_dim[1]

        # Compare the percentage of red values with the threshold
        rate = red_pixel_count / total_pixels
        if rate > self._threshold:
            return True
        else:
            return False
