import argparse
import tensorflow as tf
import cv2 as cv
import numpy as np
import torch

from cfg import cfg_slim
from prior_box import PriorBox
from utils import resize_image, resize_image_to_original, decode_landm, decode, get_head_turn, py_cpu_nms

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-i', '--image_path', default='Adrien_Brody.png')
args = parser.parse_args()


if __name__ == '__main__':
    image_path = args.image_path

    w, h, side = 320, 320, 320
    img_raw = cv.imread(image_path)
    original_height, original_width = img_raw.shape[:2]
    resized_img_raw = resize_image(img_raw, side)
    resized_img = np.float32(resized_img_raw)
    resized_img -= (104, 117, 123)
    resized_img = resized_img.transpose(2, 0, 1)
    resized_img = torch.from_numpy(resized_img).unsqueeze(0)
    img = resized_img.to("cpu")

    interpreter = tf.lite.Interpreter(model_path="slim.tflite")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    # print(input_details[0])
    # print(output_details[0])
    # print(output_details[1])
    # print(output_details[2])

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data0 = interpreter.get_tensor(output_details[0]['index'])  # loc
    output_data1 = interpreter.get_tensor(output_details[1]['index'])  # landms
    output_data2 = interpreter.get_tensor(output_details[2]['index'])  # conf

    # scores from conf
    scores = output_data2.squeeze(0)
    scores = scores[:, 1]

    # boxes from loc
    priorbox = PriorBox(cfg_slim, image_size=(w, h))
    priors = priorbox.forward()
    priors = priors.to("cpu")
    prior_data = priors.data
    loc_torch = torch.from_numpy(output_data0.squeeze(0))
    boxes = decode(loc_torch, prior_data, cfg_slim['variance'])
    scale = torch.Tensor([w, h, w, h])
    scale = scale.to("cpu")
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()

    # landmarks
    scale1 = torch.Tensor([w, h, w, h, w, h, w, h, w, h])
    landms_torch = torch.from_numpy(output_data1.squeeze(0))
    landms = decode_landm(landms_torch, prior_data, cfg_slim['variance'])
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    confidence_threshold = 0.02
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    top_k = 5000
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    nms_threshold = 0.4
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    keep_top_k = 750
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    scale_factor = max(resized_img_raw.shape) / side
    boxes = boxes * scale_factor
    landms = landms * scale_factor

    dets = np.concatenate((dets, landms), axis=1)

    visualization_threshold = 0.8
    for b in dets:
        if b[4] < visualization_threshold:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv.rectangle(resized_img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv.putText(resized_img_raw, text, (cx, cy), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv.circle(resized_img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)  # right eye
        cv.circle(resized_img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)  # left eye
        cv.circle(resized_img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)  # nose
        # cv.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        # cv.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

        # head turn
        head_turn = get_head_turn(b)
        cv.putText(resized_img_raw, head_turn, (cx, cy - 15), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    # save image
    name = "result.png"
    resized_image_back_to_original = resize_image_to_original(resized_img_raw, original_width,original_height)
    cv.imwrite(name, resized_image_back_to_original)