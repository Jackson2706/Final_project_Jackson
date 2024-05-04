from datetime import datetime
import numpy as np
import cv2

from aicore.detection.yolo_onnx.yolo_onnx import YoloOnnxDetector
from aicore.component.Trajectory import Trajectory

class Yolov5_Onnx(YoloOnnxDetector):
    def __init__(self, onnx_model_path: str, input_shape: tuple, confidence_threshold: float, nms_threshold: float, label_list: dict):
        super().__init__(onnx_model_path, input_shape, confidence_threshold, nms_threshold, label_list)

    def _preprocessing(self, frame):
        self.img_height, self.img_width = frame.shape[:2]

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
        grid_size = out_data.shape[1]

        x_center = out_data[0, :, 0]
        y_center = out_data[0, :, 1]
        w = out_data[0, :, 2]
        h = out_data[0, :, 3]
        confidence = out_data[0, :, 4]
        class_probs = out_data[0, :, 5:]

        class_id = np.argmax(class_probs, axis=1)
        class_prob = np.max(class_probs * confidence[:, np.newaxis], axis=1)

        # Filter detections based on confidence threshold
        mask = class_prob > self.confidence_threshold
        detections = [
            Trajectory(
                x_center=x_center[i],
                y_center=y_center[i],
                width=w[i],
                height=h[i],
                label = self.names[class_id[i]],
                conf=class_prob[i],
                time_stamp=datetime.now()
            ) for i in range(len(mask)) if mask[i]
        ]

        if detections:
            boxes = np.array([detection.get_bounding_box() for detection in detections])
            confidences = np.array([detection.get_confidence_score() for detection in detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), self.confidence_threshold, self.nms_threshold)
            if indices.size > 0:
                indices = indices.flatten()
                detections = [detections[i] for i in indices]
            else:
                detections = []
        
        return detections

    
    def drawbox(self, frame, results):
        image_draw = cv2.resize(frame, self.input_shape)
        for trajectory in results:
            x_center, y_center, w, h, lbl, conf, timestamp = trajectory.get_trajectory()
            x = x_center - w/2
            y = y_center - h/2
            x_max = x_center + w/2
            y_max = y_center + h/2
            class_name = lbl
            cv2.rectangle(image_draw, (int(x), int(y)), (int(x_max), int(y_max)), (0, 255, 0), 1)
            cv2.putText(image_draw, class_name, (int(x), int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255],
                    thickness=2)
            
        return image_draw