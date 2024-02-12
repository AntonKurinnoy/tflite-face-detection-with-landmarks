import math
import cv2 as cv
import numpy as np
import torch



def resize_image_to_original(img, original_width, original_height):
    if img is None:
        raise ValueError("Could not read image")

    original_aspect_ratio = original_width / original_height
    side_length = max(original_width, original_height)
    resized_image = cv.resize(img, (side_length, side_length), interpolation=cv.INTER_AREA)

    if original_aspect_ratio > 1:
        new_width = side_length
        new_height = int(new_width / original_aspect_ratio)
        y_offset = (side_length - new_height) // 2
        x_offset = 0
    else:
        new_height = side_length
        new_width = int(new_height * original_aspect_ratio)
        x_offset = (side_length - new_width) // 2
        y_offset = 0

    cropped_image = resized_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width]
    final_image = cv.resize(cropped_image, (original_width, original_height), interpolation=cv.INTER_AREA)

    return final_image


def resize_image(img, side):
    if img is None:
        raise ValueError(f"Could not read image at {img}")

    height, width = img.shape[:2]
    aspect_ratio = width / height

    if width > height:
        new_width = side
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = side
        new_width = int(new_height * aspect_ratio)

    resized_image = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)
    square_image = np.zeros((side, side, 3), dtype=np.uint8)

    x_offset = (side - new_width) // 2
    y_offset = (side - new_height) // 2

    square_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

    return square_image


def get_head_turn(landmarks):
    a = (landmarks[5], landmarks[6])  # right eye
    b = (landmarks[7], landmarks[8])  # left eye
    c = (landmarks[9], landmarks[10])  # nose
    result = ""

    abc = get_angle(a, b, c)
    bac = get_angle(b, a, c)
    acb = get_angle(a, c, b)
    # print("Angle(abc) between vectors BA and BC: {} degrees".format(abc))
    # print("Angle(bac) between vectors AB and AC: {} degrees".format(bac))
    # print("Angle(acb) between vectors AC and BC: {} degrees".format(acb))

    if 75 < acb < 90:
        result = result + "FRONT"
    elif acb > 90:
        result = result + "UP"
    elif acb < 75:
        result = result + "DOWN"

    if abs(abc - bac) < 5:
        result = result + "-FRONT"
    elif abc > bac:
        result = result + "-LEFT"
    else:
        result = result + "-RIGHT"

    return result


def get_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0] * ba[0] + ba[1] * ba[1])
    magnitude_bc = math.sqrt(bc[0] * bc[0] + bc[1] * bc[1])

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    angle = math.acos(cosine_angle)

    return math.degrees(angle)


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
