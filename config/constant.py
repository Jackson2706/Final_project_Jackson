### SYSTEM CONFIG
VERSION = 1
SERVER_IP = "http://localhost:8080/Aithings-camAi/api/v{}/".format(VERSION)
SERIAL = "662fd3484858db519a0d5f36"
USERNAME = "dev"
PASSWORD = "1"
CAM_ID = "CAM04"
# CAMERA_URL = "rtsp://admin:Admin@123@27.72.149.50:1554/profile3/media.smp"
# CAMERA_URL = 0
CAMERA_URL = "Video test.mp4"
MAX_FRAME_QUEUE_SIZE = 500
SELECTED_CATEGORY = [
    "bicycle", "car", "motorcycle", "bus", "truck"
]
SELECTED_DETECTOR_NAME = "YOLOV5_ONNX" #params: [YOLOV5_ONNX, YOLOV5_TRT, SSD300_ONNX]
SELECTED_TRACKER_NAME = ""

### YOLO COCO TENSORT CONFIG
TENSORRT_LIB_PATH = "aicore/detection/yolo_trt/yolov5/config/libmyplugins.so"
TENSORRT_MODEL_ENGINE = "aicore/detection/yolo_trt/yolov5/config/yolov5m.engine"
YOLOV5_TENSORRT_LABEL_LIST = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

YOLOV5_TENSORRT_CONF_THRESH = 0.4
YOLOV5_TENSORRT_IOU_THRESHOLD = 0.4
YOLOV5_TENSORRT_LEN_ALL_RESULT = 38001
YOLOV5_TENSORRT_LEN_ONE_RESULT = 38

### YOLOV5 COCO ONNX CONFIG
YOLOV5_ONNX_MODEL_PATH = "aicore/detection/yolo_onnx/yolov5/config/yolov5s.onnx"
YOLOV5_ONNX_INPUT_SHAPE = (640, 640)
YOLOV5_ONNX_LABEL_LIST = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]
YOLOV5_ONNX_CONF_THRESHOLD = 0.4
YOLOV5_ONNX_IOU_THRESHOLD = 0.4

### SSD300 ONNX CONFIG
SSD300_ONNX_MODEL_PATH = "aicore/detection/SSD_onnx/SSD300/config/ssd300.onnx"
SSD300_ONNX_INPUT_SHAPE = (300, 300)
SSD300_ONNX_LABEL_LIST = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
SSD300_ONNX_CONF_THRESHOLD = 0.4
SSD300_ONNX_IOU_THRESHOLD = 0.4

### SSD320 ONNX CONFIG
SSD320_ONNX_MODEL_PATH = "aicore/detection/SSD_onnx/SSD300/config/ssd300.onnx"
SSD320_ONNX_INPUT_SHAPE = (300, 300)
SSD320_ONNX_LABEL_LIST = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
SSD320_ONNX_CONF_THRESHOLD = 0.4
SSD320_ONNX_IOU_THRESHOLD = 0.4

### DEEPSORT CONFIG
DEEPSORT_CONFIG_PATH = "aicore/tracking/deep_sort/configs/deep_sort.yaml"




