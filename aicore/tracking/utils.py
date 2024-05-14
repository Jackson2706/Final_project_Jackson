import cv2
def draw_bboxes(image, bboxes, line_thickness=None):
    """
    Vẽ bounding box và tâm của bounding box trên ảnh đầu vào.

    Tham số:
    - image: Ảnh đầu vào.
    - bboxes: Danh sách các bounding box trong định dạng (x, y, w, h, label).
    - line_thickness: Độ dày của đường viền bounding box.

    Trả về:
    - image: Ảnh với bounding box và tâm được vẽ lên.
    """
    line_thickness = line_thickness or int(
        0.002 * (image.shape[0] + image.shape[1]) * 0.5) + 1

    for i, bbox in bboxes.items():
        x, y, w, h, _, conf, _ = bbox.get_last_trajectory().get_trajectory()
        label = bbox.get_label()
        track_id = bbox.get_track_id()
        color = (0, 255, 0)
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        check_point_x = x1
        check_point_y = int(y1 + ((y2 - y1) * 0.6))
        c1, c2 = (x1, y1), (x2, y2)

        # Vẽ bounding box
        cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

        # Vẽ tâm của bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(image, (center_x, center_y), radius=2, color=(255, 0, 0), thickness=-1)  # Vẽ hình tròn

        # Vẽ tên nhãn
        font_thickness = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize("{} - ID: {}".format(label, track_id), 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # Đổ màu
        cv2.putText(image, "{} - ID: {}".format(label, track_id), (c1[0], c1[1] - 2), 0, line_thickness / 3,
                    [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

    return image
