from aicore.detection.yolo_onnx.yolo_onnx import YoloOnnxDetector

class Yolov5_Onnx(YoloOnnxDetector):
    def __init__(self, onnx_model_path: str, input_shape: tuple, confidence_threshold: float, nms_threshold: float, label_list: dict):
        pass