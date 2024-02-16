import math
import torch


def get_head_turn(landmarks):
    a = (landmarks[5], landmarks[6])  # right eye
    b = (landmarks[7], landmarks[8])  # left eye
    c = (landmarks[9], landmarks[10])  # nose
    result = "HEAD TURN: "

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
