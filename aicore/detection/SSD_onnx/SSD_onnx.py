from abc import ABC
import onnxruntime
from typing import Tuple, List


class SSDDetector(ABC):
    def __init__(self, ssd_onnx_model_path: str, input_shape: Tuple, confidence_threshold: float, nms_threshold: float,
                 label_list: List, selected_categories: List):
        self.session = onnxruntime.InferenceSession(ssd_onnx_model_path)
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.label_list = label_list
        self.selected_categories = selected_categories

    def _preprocessing(self, frame):
        pass

    def _postprocessing(self, out_data, timestamps):
        pass

    def _execute(self, in_data):
        return self.session.run(
            None,
            {self.session.get_inputs()[0].name: in_data}
        )

    def run_native_inference(self, frame, timestamps):
        input_data = self._preprocessing(frame=frame)
        output_data = self._execute(in_data=input_data)
        results = self._postprocessing(out_data=output_data, timestamps=timestamps)
        return results

    def draw_box(self, frame, results):
        pass
