from aicore.detection.yolo_onnx.yolo_onnx import YoloOnnxDetector

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
        pass