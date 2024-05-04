import cv2
import numpy as np
import pycuda.driver as cuda
from datetime import datetime

from aicore.detection.yolo_trt.yolo_trt import YoloVehicleDetector
from aicore.component.Trajectory import Trajectory


class Yolov5_TRT(YoloVehicleDetector):
    def __init__(self, library, engine, categories, conf_thres, nms_thres, len_all_result, len_one_result):
        """
        Initializes the YOLOv5 TensorRT object.

        Parameters:
        - library: Path to the library.
        - engine: Path to the engine file.
        - categories: List of categories.
        - conf_thres: Confidence threshold.
        - nms_thres: Non-maximum suppression threshold.
        - len_all_result: Length of all results.
        - len_one_result: Length of one result.
        """
        super().__init__(library, engine, categories, conf_thres, nms_thres, len_all_result, len_one_result)
    
    def preprocess_image(self, img):
        """
        Preprocesses an image for YOLOv5 TensorRT inference.

        Parameters:
        - img: Input image.

        Returns:
        - image: Preprocessed image.
        - image_raw: Original image.
        - h: Original image height.
        - w: Original image width.
        """
        super().preprocess_image(img)
        image_raw = img
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128))
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w
    
    def postprocess(self, output, origin_h, origin_w):
        """
        Postprocesses the output of YOLOv5 TensorRT inference.

        Parameters:
        - output: Model output.
        - origin_h: Original image height.
        - origin_w: Original image width.

        Returns:
        - result_boxes: Detected bounding boxes.
        - result_scores: Confidence scores of detected boxes.
        - result_classid: Class IDs of detected boxes.
        """
        super().postprocess(output, origin_h, origin_w)
        num = int(output[0])
        pred = np.reshape(output[1:], (-1, self.LEN_ONE_RESULT))[:num, :]
        pred = pred[:, :6]
        boxes = self._NMS(pred, origin_h, origin_w)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid
    
    def run_native_inference(self, image):
        """
        Runs native inference for YOLOv5 TensorRT.

        Parameters:
        - image: Input image.

        Returns:
        - boxes: Detected bounding boxes.
        """
        super().run_native_inference(image)
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(img=image)
        np.copyto(self.host_inputs[0], input_image.ravel())
        stream = cuda.Stream()
        self.context = self.engine.create_execution_context()
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], stream)
        self.context.execute_async(self.batch_size, self.bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], stream)
        stream.synchronize()
        output = self.host_outputs[0]
                
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.postprocess(output[i * self.LEN_ALL_RESULT: (i + 1) * self.LEN_ALL_RESULT], origin_h, origin_w)
            
        boxes = []
        for j in range(len(result_boxes)):
            box = result_boxes[j]
            lbl = self.categories[int(result_classid[j])]
            conf = result_scores[j]
            x1,y1,x2,y2 = box
            x_center = (x1+x2)/2
            y_center = (y1+y2)/2
            w = x2 - x1
            h = y2 - y1
            boxes.append(Trajectory(x_center=x_center, y_center=y_center, width=w, height=h, label=lbl, conf=conf, time_stamp=datetime.now()))
        return boxes
