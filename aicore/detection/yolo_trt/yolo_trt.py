
from abc import ABC
import numpy as np
import ctypes
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class YoloVehicleDetector(ABC):
    def __init__(self, library, engine, categories, conf_thres, nms_thres, len_all_result, len_one_result):
        """
        Initializes the YOLO Vehicle Detector.

        Parameters:
        - library: Path to the library.
        - engine: Path to the engine file.
        - categories: List of categories.
        - conf_thres: Confidence threshold.
        - nms_thres: Non-maximum suppression threshold.
        - len_all_result: Length of all results.
        - len_one_result: Length of one result.
        """
        self.library = library
        self.engine = engine
        self.categories = categories
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.LEN_ALL_RESULT = len_all_result
        self.LEN_ONE_RESULT = len_one_result
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        ctypes.CDLL(library)

        with open(engine, 'rb') as f:
            serialized_engine = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.batch_size = self.engine.max_batch_size

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.input_w = self.engine.get_binding_shape(binding)[-1]
                self.input_h = self.engine.get_binding_shape(binding)[-2]
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

    def check_gpu(self):
        """
        Checks if the code is running on GPU.

        Returns:
        - gpu_available: True if GPU is available, False otherwise.
        """
        gpu_available = cuda.Device.count() > 0
        if not gpu_available:
            print("No GPU device found. Running on CPU.")
        return gpu_available
    
    def preprocess_image(self, img):
        """
        Preprocesses an image.

        Parameters:
        - img: Input image.

        Returns:
        - preprocessed_image: Preprocessed image.
        """
        pass

    def run_native_inference(self, image):
        """
        Runs native inference.

        Parameters:
        - image: Input image.

        Returns:
        - result: Inference result.
        """
        pass

    def postprocess(self, output, origin_h, origin_w):
        """
        Postprocesses the output.

        Parameters:
        - output: Model output.
        - origin_h: Original image height.
        - origin_w: Original image width.

        Returns:
        - processed_output: Processed output.
        """
        pass
    
    def _NMS(self, prediction, origin_h, origin_w):
        """
        Performs Non-Maximum Suppression (NMS).

        Parameters:
        - prediction: Model prediction.
        - origin_h: Original image height.
        - origin_w: Original image width.

        Returns:
        - boxes: Detected bounding boxes after NMS.
        """
        boxes = prediction[prediction[:, 4] >= self.conf_thres]
        boxes[:, :4] = self._xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        confs = boxes[:, 4]
        boxes = boxes[np.argsort(-confs)]
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self._bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > self.nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes
    
    def _bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        Calculates Intersection over Union (IoU) of two bounding boxes.

        Parameters:
        - box1: First bounding box.
        - box2: Second bounding box.
        - x1y1x2y2: Format of bounding box coordinates.

        Returns:
        - iou: Intersection over Union (IoU) value.
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def _xywh2xyxy(self, origin_h, origin_w, x):
        """
        Converts bounding box coordinates from (x, y, w, h) to (x1, y1, x2, y2) format.

        Parameters:
        - origin_h: Original image height.
        - origin_w: Original image width.
        - x: Bounding box coordinates.

        Returns:
        - y: Converted bounding box coordinates.
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
        return y