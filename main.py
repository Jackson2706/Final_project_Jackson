import cv2

from application.streaming.streaming import IPCameraFrameRecoder
from config.constant import CAMERA_URL, MAX_FRAME_QUEUE_SIZE, TENSORRT_LIB_PATH, TENSORRT_MODEL_ENGINE, CATEGORY, LEN_ALL_RESULT, LEN_ONE_RESULT, IOU_THRESHOLD, CONF_THRESH, ONNX_MODEL_PATH, INPUT_SHAPE, DEEPSORT_CONFIG_PATH
from aicore.detection.yolo_onnx.yolov5.yolov5_onnx import Yolov5_Onnx
from aicore.tracking.deep_sort.tracker import DeepSortInference, draw_bboxes

cap = IPCameraFrameRecoder(camera_url=CAMERA_URL,
                           max_queue_size=MAX_FRAME_QUEUE_SIZE,
                           frame_dir="./frame_dir",
                           log_file="./streaming_loger.txt")

vehicle_detector = Yolov5_Onnx(onnx_model_path=ONNX_MODEL_PATH,
                                    input_shape=INPUT_SHAPE,
                                    confidence_threshold=CONF_THRESH,
                                    nms_threshold=IOU_THRESHOLD,
                                    label_list={index: value for index, value in enumerate(CATEGORY)})
vehicle_tracker = DeepSortInference(config_path=DEEPSORT_CONFIG_PATH)
for frame in cap.read_frame():
    result = vehicle_detector.run_native_inference(frame)
    result = vehicle_tracker.update(bboxes_traject=result, image=frame)
    frame = cv2.resize(frame, INPUT_SHAPE)
    frame = draw_bboxes(image=frame, bboxes=result)
    cv2.imshow("test", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()