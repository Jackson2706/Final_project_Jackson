from abc import ABC
import onnxruntime

class YoloOnnxDetector(ABC):
    def __init__(self, onnx_model_path: str, input_shape: tuple, confidence_threshold: float, nms_threshold: float, label_list: dict):
        """
        Initialize the YoloOnnxDetector class with required parameters.

        Args:
        - onnx_model_path (str): Path to the ONNX model file.
        - input_shape (tuple): Shape of the input image (height, width).
        - confidence_threshold (float): Minimum confidence threshold for detections.
        - nms_threshold (float): Non-maximum suppression threshold.
        - label_list (dict): Dictionary mapping class indices to class names.
        """
        # Create an inference session with the ONNX model
        self.session = onnxruntime.InferenceSession(onnx_model_path)
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.names = label_list

    def _preprocessing(self, frame):
        """
        Preprocess the input frame before inference.

        Args:
        - frame: Input frame to preprocess.
        """
        pass

    def _execute(self, input_data):
        """
        Execute the inference on the input data.

        Args:
        - input_data: Input data for the inference.

        Returns:
        - out_data: Output data from the inference.
        """
        return self.session.run(
            None,
            {self.session.get_inputs()[0].name: input_data}
        )[0]

    def _postprocessing(self, out_data):
        """
        Postprocess the output data from inference.

        Args:
        - out_data: Output data from the inference.
        """
        pass

    def run_native_inference(self, frame):
        """
        Run native inference on the input frame.

        Args:
        - frame: Input frame for inference.
        """
        pass

    def drawbox(self, frame, results): 
        """
        Draw bounding boxes on the input frame based on the results.

        Args:
        - frame: Input frame to draw bounding boxes on.
        - results: Results from the inference to draw bounding boxes.
        """
        pass
