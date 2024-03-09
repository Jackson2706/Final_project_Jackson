

class Deepsort:
    def __init__(self, model_path,  max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        pass

    def update(self, bbox_xywh, confidences, ori_img):
        pass

    def _tlwh_to_xyxy(self, bbox_tlwh):
       pass

    def _xyxy_to_tlwh(self, bbox_xyxy):
        pass
    
    def _get_features(self, bbox_xywh, ori_img):
        pass