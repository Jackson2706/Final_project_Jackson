from typing import Tuple, List
import cv2
import numpy as np
from datetime import datetime

from aicore.detection.SSD_onnx.SSD_onnx import SSDDetector
from aicore.component.Trajectory import Trajectory


class SSD320OnnxDetector(SSDDetector):
    def __init__(self, ssd_onnx_model_path: str, input_shape: Tuple, confidence_threshold: float, nms_threshold: float,
                 label_list: List, selected_categories: List):
        super().__init__(ssd_onnx_model_path, input_shape, confidence_threshold, nms_threshold, label_list,
                         selected_categories)

    def _preprocessing(self, frame):
        original_height, original_width = frame.shape[:2]
        self.resize_ratio_w = original_width / self.input_shape[0]
        self.resize_ratio_h = original_height / self.input_shape[1]

        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, self.input_shape)

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def _postprocessing(self, out_data):
        detections = []
        bbox_list, conf_list, label_list = out_data
        for box, conf, label in zip(bbox_list, conf_list, label_list):
            if conf < self.confidence_threshold or self.label_list[label] not in self.selected_categories:
                continue
            x_min, y_min, x_max, y_max = box
            detections.append(
                Trajectory(
                    x_center=(x_min + x_max) / 2 * self.resize_ratio_w,
                    y_center=(y_min + y_max) / 2 * self.resize_ratio_h,
                    width=(x_max - x_min) * self.resize_ratio_w,
                    height=(y_max - y_min) * self.resize_ratio_h,
                    label=self.label_list[label],
                    conf=conf,
                    time_stamp=datetime.now()
                )
            )
        if detections:
            boxes = np.array([detection.get_bounding_box() for detection in detections])
            confidences = np.array([detection.get_confidence_score() for detection in detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), self.confidence_threshold,
                                       self.nms_threshold)
            if indices.size > 0:
                indices = indices.flatten()
                detections = [detections[i] for i in indices]
            else:
                detections = []

        return detections
