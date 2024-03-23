import cv2
import torch
import numpy as np
import sys
from aicore.tracking.deep_sort.utils.parser import get_config
from aicore.tracking.deep_sort.deep_sort.deep_sort import DeepSort
from aicore.component.detected_object import Detected_Object
from aicore.component.trajectory import Trajectory

class DeepSortInference:
    def __init__(self, config_path:str):
        """
        Initialize the DeepSortInference object with the given configuration file path.

        Parameters:
        - config_path (str): The path to the DeepSort configuration file.
        """
        cfg = get_config()
        cfg.merge_from_file(config_path)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        
        self.bboxes2draw = {}
        self.capacity = 1000

    def draw_bboxes(self,image, bboxes, line_thickness=None):
        """
        Draw bounding boxes on the given image.

        Parameters:
        - image: The input image.
        - bboxes: A list of bounding boxes in the format (x1, y1, x2, y2, class_id, pos_id).
        - line_thickness: The thickness of the bounding box lines.

        Returns:
        - image: The image with bounding boxes drawn on it.
        """
        line_thickness = line_thickness or int(
            0.002 * (image.shape[0] + image.shape[1]) * 0.5) + 1

        list_pts = []
        point_radius = 4
        for bbox in bboxes:
            x,y,w,h,label, conf, _ = bbox.get_trajectory()
            color = (0, 255, 0)
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            # 撞线的点
            check_point_x = x1
            check_point_y = int(y1 + ((y2 - y1) * 0.6))
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

            font_thickness = max(line_thickness - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, ' ID-{}'.format(label), (c1[0], c1[1] - 2), 0, line_thickness / 3,
                        [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

            list_pts.append([check_point_x - point_radius, check_point_y - point_radius])
            list_pts.append([check_point_x - point_radius, check_point_y + point_radius])
            list_pts.append([check_point_x + point_radius, check_point_y + point_radius])
            list_pts.append([check_point_x + point_radius, check_point_y - point_radius])

            ndarray_pts = np.array(list_pts, np.int32)

            cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))

            list_pts.clear()

        return image


    def update(self,bboxes_traject, image):
        """
        Perform DeepSort update with the given bounding boxes and image.

        Parameters:
        - bboxes_traject: A list of bounding boxes with associated trajectories.
        - image: The input image.

        Returns:
        - bboxes2draw: A dictionary containing updated Detected_Object instances.
        """
        min_track_id = sys.maxsize - 1
        bbox_xywh = []
        confs = []
        time_stamp_list = []
        bboxes = [bbox.get_trajectory() for bbox in bboxes_traject]
        if len(bboxes) > 0:
            for x, y, w, h, lbl, conf, timestamp in bboxes:
                obj = [x, y, w, h]
                bbox_xywh.append(obj)
                confs.append(conf)
                time_stamp_list.append(timestamp)
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)
            outputs = self.deepsort.update(xywhs, confss, time_stamp_list,image)

            for center_x, center_y, w, h, track_id, conf, timestamp in list(outputs):
                label = self.search_label(center_x=center_x, center_y=center_y,
                                    bboxes_xyxy=bboxes, max_dist_threshold=20.0)
                center_x = int(center_x)
                center_y = int(center_y)
                w = int(w)
                h = int(h)
                if (min_track_id > track_id):
                    min_track_id = track_id

                if track_id not in self.bboxes2draw:
                    self.bboxes2draw[track_id] = Detected_Object(track_id=track_id, label=label, trajectories=[Trajectory(x_center=center_x, y_center=center_y, width=w, height=h, label=label, conf=conf, time_stamp=timestamp)])
                else:
                    self.bboxes2draw[track_id].add_trajectory(Trajectory(x_center=center_x, y_center=center_y, width=w, height=h, label=label, conf=conf, time_stamp=timestamp))
        if(len(self.bboxes2draw) > self.capacity):
            self.bboxes2draw = {key: value for key, value in self.bboxes2draw.items() if key < min_track_id}
        return self.bboxes2draw


    def search_label(self, center_x, center_y, bboxes_xyxy, max_dist_threshold):
        """
        Search for the label closest to the center point within the given bounding boxes.

        Parameters:
        - center_x: X-coordinate of the center point.
        - center_y: Y-coordinate of the center point.
        - bboxes_xyxy: List of bounding boxes in format (x1, y1, x2, y2, label, confidence, _).
        - max_dist_threshold: Maximum distance threshold for considering a match.

        Returns:
        - label: The label string.
        """
        label = ''
        min_dist = -1.0

        for center_x2, center_y2, w, h, lbl, conf, _ in bboxes_xyxy:
            min_x = abs(center_x2 - center_x)
            min_y = abs(center_y2 - center_y)

            if min_x < max_dist_threshold and min_y < max_dist_threshold:
                avg_dist = (min_x + min_y) * 0.5
                if min_dist == -1.0:
                    min_dist = avg_dist
                    label = lbl
                    pass
                else:
                    if avg_dist < min_dist:
                        min_dist = avg_dist
                        label = lbl
                    pass
                pass
            pass

        return label

