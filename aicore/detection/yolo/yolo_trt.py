
from abc import ABC

class YoloVehicleDetector(ABC):
    def __init__(self, library, engine, origin_h, origin_w, conf_thres, nms_thres):
        pass
    def preprocess_image(self, image):
        pass
    def run_native_inference(self, image):
        pass
    def postprocess(self, output):
        pass
    def NMS(self, prediction):
        pass
    def bbox_iou(self, box1, box2, x1y1,x2y2=True):
        pass
    def plot_box(self, x, img, color, label=None, line_thickness=None):
        pass