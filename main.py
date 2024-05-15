import cv2
from threading import Thread
from datetime import datetime
from application.streaming.streaming import IPCameraFrameRecoder
from config.constant import CAMERA_URL, MAX_FRAME_QUEUE_SIZE, YOLOV5_ONNX_MODEL_PATH, YOLOV5_ONNX_INPUT_SHAPE, \
    YOLOV5_ONNX_CONF_THRESHOLD, YOLOV5_ONNX_IOU_THRESHOLD, YOLOV5_ONNX_LABEL_LIST, DEEPSORT_CONFIG_PATH, \
    SELECTED_CATEGORY, \
    SELECTED_DETECTOR_NAME, SELECTED_TRACKER_NAME, SSD300_ONNX_MODEL_PATH, SSD300_ONNX_INPUT_SHAPE, \
    SSD300_ONNX_LABEL_LIST, \
    SSD300_ONNX_CONF_THRESHOLD, SSD300_ONNX_IOU_THRESHOLD, SSD320_ONNX_MODEL_PATH, SSD320_ONNX_IOU_THRESHOLD, \
    SSD320_ONNX_CONF_THRESHOLD, \
    SSD320_ONNX_INPUT_SHAPE, SSD320_ONNX_LABEL_LIST, TENSORRT_LIB_PATH, TENSORRT_MODEL_ENGINE, \
    YOLOV5_TENSORRT_LEN_ALL_RESULT, YOLOV5_TENSORRT_LEN_ONE_RESULT

from aicore.detection.yolo_onnx.yolov5.yolov5_onnx import Yolov5_Onnx
from aicore.detection.SSD_onnx.SSD300.SSD300_onnx_detector import SSD300OnnxDetector
from aicore.detection.SSD_onnx.SSDLite320.SSD320_onnx_detector import SSD320OnnxDetector
# from aicore.detection.yolo_trt.yolov5.yolov5_trt import Yolov5_TRT
from aicore.tracking.deep_sort.tracker import DeepSortInference
from aicore.tracking.kalman_filter.KalmanFilterTracking import KalmanFilterTracking
from aicore.tracking.utils import draw_bboxes
## Test
from aicore.logic_shape_filter.filter.LineFilter import LineFilter
from aicore.component.Line import Line
from aicore.component.Point import Point
from utils import ViolationVideoSaver

line_location = Line(Point(50, 1500), Point(2000, 1500))
linefilter = LineFilter("Test_line", 1, line_location, 0, 0, ["car"], 0)
filter_dict = {1: linefilter}

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
# elif SELECTED_DETECTOR_NAME == "YOLOV5_TRT":
#     vehicle_detector = Yolov5_TRT(library=TENSORRT_LIB_PATH,
#                                   engine=TENSORRT_MODEL_ENGINE,
#                                   categories=YOLOV5_ONNX_LABEL_LIST,
#                                   conf_thres=YOLOV5_ONNX_CONF_THRESHOLD,
#                                   nms_thres=YOLOV5_ONNX_IOU_THRESHOLD,
#                                   len_all_result=YOLOV5_TENSORRT_LEN_ALL_RESULT,
#                                   len_one_result=YOLOV5_TENSORRT_LEN_ONE_RESULT)
elif SELECTED_DETECTOR_NAME == "SSD300_ONNX":
    vehicle_detector = SSD300OnnxDetector(ssd_onnx_model_path=SSD300_ONNX_MODEL_PATH,
                                          input_shape=SSD300_ONNX_INPUT_SHAPE,
                                          confidence_threshold=SSD300_ONNX_CONF_THRESHOLD,
                                          nms_threshold=SSD300_ONNX_IOU_THRESHOLD,
                                          label_list=SSD300_ONNX_LABEL_LIST, selected_categories=SELECTED_CATEGORY)

elif SELECTED_DETECTOR_NAME == "SSD320_ONNX":
    vehicle_detector = SSD320OnnxDetector(ssd_onnx_model_path=SSD320_ONNX_MODEL_PATH,
                                          input_shape=SSD320_ONNX_INPUT_SHAPE,
                                          confidence_threshold=SSD320_ONNX_CONF_THRESHOLD,
                                          nms_threshold=SSD320_ONNX_IOU_THRESHOLD,
                                          label_list=SSD320_ONNX_LABEL_LIST, selected_categories=SELECTED_CATEGORY)
else:
    vehicle_detector = None
if SELECTED_TRACKER_NAME == "DEEPSORT":
    vehicle_tracker = DeepSortInference(config_path=DEEPSORT_CONFIG_PATH)
else:
    vehicle_tracker = KalmanFilterTracking(150)


video_saver_dict = {}
visited_id = {}
recording_id = []
## Run main
for frame, current_time in cap.read_frame():
    result = vehicle_detector.run_native_inference(frame=frame, timestamps=current_time)
    result = vehicle_tracker.update(bboxes_traject=result, image=frame)
    frame = draw_bboxes(image=frame, bboxes=result)
    for idx, filter_element in filter_dict.items():
        result = filter_element.make_filtering(result)
        frame = filter_element.draw(frame)
    if len(result):
        # create a new thread to save video about this trajeetories of vehicle
        for id, obj in result.items():
            if id not in visited_id:
                visited_id[id] = current_time
                continue
            if id in visited_id and current_time - visited_id[id] <= 4000:
                continue
            if id in recording_id:
                continue
            else:
                recording_id.append(id)
            print("Start record: {}".format(id))
            violation_video_saver = ViolationVideoSaver(save_path=f"./{id}.avi", cam_object=cap, selected_obj=obj, fiter_dic=filter_dict)
            violation_video_saver.start()
    frame = cv2.resize(frame, (900, 900))
    cv2.imshow("test", frame)

    if cv2.waitKey(1) == ord("q"):
        break

for video_saver in video_saver_dict.values():
    video_saver.stop()

cap.release()
cv2.destroyAllWindows()