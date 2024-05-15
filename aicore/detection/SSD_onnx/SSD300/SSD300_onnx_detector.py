import cv2
import numpy as np
from typing import Tuple, List
from datetime import datetime

from aicore.detection.SSD_onnx.SSD_onnx import SSDDetector
from aicore.component.Trajectory import Trajectory


class SSD300OnnxDetector(SSDDetector):

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

    def _postprocessing(self, out_data, timestamps):
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
                    time_stamp=timestamps
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

    def draw_box(self, frame, results):
        image_draw = frame
        for trajectory in results:
            x_center, y_center, w, h, lbl, conf, timestamp = trajectory.get_trajectory()
            x = x_center - w / 2
            y = y_center - h / 2
            x_max = x_center + w / 2
            y_max = y_center + h / 2
            class_name = lbl
            cv2.rectangle(image_draw, (int(x), int(y)), (int(x_max), int(y_max)), (0, 255, 0), 1)
            cv2.putText(image_draw, class_name, (int(x), int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        [225, 255, 255],
                        thickness=2)

        return image_draw


if __name__ == "__main__":
    cap = cv2.VideoCapture("/home/jackson/Desktop/Final_project_Jackson/Video test.mp4")
    detector = SSD300OnnxDetector(
        ssd_onnx_model_path="/home/jackson/Desktop/Final_project_Jackson/aicore/detection/SSD_onnx/SSD300/config/ssd300.onnx",
        input_shape=(300, 300), confidence_threshold=0.4, nms_threshold=0.4,
        label_list=['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
                    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'])
    print(detector)
    while True:
        ret, frame = cap.read()
        results = detector.run_native_inference(frame)
        frame = detector.draw_box(frame, results)
        print(results)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
