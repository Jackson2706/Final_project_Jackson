import cv2

from application.streaming.streaming import IPCameraFrameRecoder
from config.constant import CAMERA_URL, MAX_FRAME_QUEUE_SIZE, YOLOV5_ONNX_MODEL_PATH, YOLOV5_ONNX_INPUT_SHAPE, \
    YOLOV5_ONNX_CONF_THRESHOLD, YOLOV5_ONNX_IOU_THRESHOLD, YOLOV5_ONNX_LABEL_LIST, DEEPSORT_CONFIG_PATH, SELECTED_CATEGORY, \
    SELECTED_DETECTOR_NAME, SELECTED_TRACKER_NAME, SSD300_ONNX_MODEL_PATH, SSD300_ONNX_INPUT_SHAPE, SSD300_ONNX_LABEL_LIST, \
    SSD300_ONNX_CONF_THRESHOLD, SSD300_ONNX_IOU_THRESHOLD, SSD320_ONNX_MODEL_PATH, SSD320_ONNX_IOU_THRESHOLD, SSD320_ONNX_CONF_THRESHOLD, \
    SSD320_ONNX_INPUT_SHAPE, SSD320_ONNX_LABEL_LIST, TENSORRT_LIB_PATH, TENSORRT_MODEL_ENGINE,YOLOV5_TENSORRT_LEN_ALL_RESULT, YOLOV5_TENSORRT_LEN_ONE_RESULT

from aicore.detection.yolo_onnx.yolov5.yolov5_onnx import Yolov5_Onnx
from aicore.detection.SSD_onnx.SSD300.SSD300_onnx_detector import SSD300OnnxDetector
from aicore.detection.SSD_onnx.SSDLite320.SSD320_onnx_detector import SSD320OnnxDetector
from aicore.detection.yolo_trt.yolov5.yolov5_trt import Yolov5_TRT
from aicore.tracking.deep_sort.tracker import DeepSortInference
from aicore.tracking.kalman_filter.KalmanFilterTracking import KalmanFilterTracking
from aicore.tracking.utils import draw_bboxes
## Test
from aicore.logic_shape_filter.filter.LineFilter import LineFilter
from aicore.component.Line import Line
from aicore.component.Point import Point

line_location = Line(Point(45, 411), Point(551, 455))
linefilter = LineFilter("Test_line", 1, line_location, 0, 0, ["car"], -1)
####

cap = IPCameraFrameRecoder(camera_url=CAMERA_URL,
                           max_queue_size=MAX_FRAME_QUEUE_SIZE,
                           frame_dir="./frame_dir",
                           log_file="./streaming_loger.txt")

if SELECTED_DETECTOR_NAME == "YOLOV5_ONNX":
    vehicle_detector = Yolov5_Onnx(onnx_model_path=YOLOV5_ONNX_MODEL_PATH,
                               input_shape=YOLOV5_ONNX_INPUT_SHAPE,
                               confidence_threshold=YOLOV5_ONNX_CONF_THRESHOLD,
                               nms_threshold=YOLOV5_ONNX_IOU_THRESHOLD,
                               label_list=YOLOV5_ONNX_LABEL_LIST,
                               selected_class=SELECTED_CATEGORY)
elif SELECTED_DETECTOR_NAME == "YOLOV5_TRT":
    vehicle_detector = Yolov5_TRT(library=TENSORRT_LIB_PATH,
                                  engine=TENSORRT_MODEL_ENGINE,
                                  categories=YOLOV5_ONNX_LABEL_LIST,
                                  conf_thres=YOLOV5_ONNX_CONF_THRESHOLD,
                                  nms_thres=YOLOV5_ONNX_IOU_THRESHOLD,
                                  len_all_result=YOLOV5_TENSORRT_LEN_ALL_RESULT,
                                  len_one_result=YOLOV5_TENSORRT_LEN_ONE_RESULT)
elif SELECTED_DETECTOR_NAME == "SSD300_ONNX":
    vehicle_detector = SSD300OnnxDetector(ssd_onnx_model_path=SSD300_ONNX_MODEL_PATH, input_shape=SSD300_ONNX_INPUT_SHAPE,
                                          confidence_threshold=SSD300_ONNX_CONF_THRESHOLD, nms_threshold=SSD300_ONNX_IOU_THRESHOLD,
                                          label_list=SSD300_ONNX_LABEL_LIST, selected_categories=SELECTED_CATEGORY)

elif SELECTED_DETECTOR_NAME == "SSD320_ONNX":
    vehicle_detector = SSD320OnnxDetector(ssd_onnx_model_path=SSD320_ONNX_MODEL_PATH, input_shape=SSD320_ONNX_INPUT_SHAPE,
                                          confidence_threshold=SSD320_ONNX_CONF_THRESHOLD, nms_threshold=SSD320_ONNX_IOU_THRESHOLD,
                                          label_list=SSD320_ONNX_LABEL_LIST, selected_categories=SELECTED_CATEGORY)
else:
    vehicle_detector = None
if SELECTED_TRACKER_NAME == "DEEPSORT":
    vehicle_tracker = DeepSortInference(config_path=DEEPSORT_CONFIG_PATH)
else:
    vehicle_tracker = KalmanFilterTracking(20)
for frame in cap.read_frame():
    result = vehicle_detector.run_native_inference(frame)
    result = vehicle_tracker.update(bboxes_traject=result, image=frame)
    frame = draw_bboxes(image=frame, bboxes=result)
    frame = cv2.resize(frame, (900, 900))

    cv2.imshow("test", frame)
    filter_result = linefilter.make_filtering(result)
    # print(len(filter_result))
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
