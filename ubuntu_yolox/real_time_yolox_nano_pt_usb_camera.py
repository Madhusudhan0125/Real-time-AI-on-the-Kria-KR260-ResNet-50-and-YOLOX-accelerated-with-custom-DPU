#!/usr/bin/env python
# -*- coding: utf-8 -*-

print("\nRunning YOLOX on camera feed using DPU-PYNQ\n")

# ***********************************************************************
# Imports
# ***********************************************************************
import os
import time
import numpy as np
import cv2
import random
import colorsys
from pynq_dpu import DpuOverlay

# ***********************************************************************
# File paths
# ***********************************************************************
dpu_model   = os.path.abspath("dpu.bit")
cnn_xmodel  = os.path.join("./", "yolox_nano_pt.xmodel")
labels_file = os.path.join("./img", "coco2017_classes.txt")

# ***********************************************************************
# Load DPU overlay and model
# ***********************************************************************
overlay = DpuOverlay(dpu_model)
overlay.load_model(cnn_xmodel)
dpu = overlay.runner

# ***********************************************************************
# Helper functions
# ***********************************************************************
def preprocess(image, input_size):
    padded_image = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    ratio = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    resized_image = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
    padded_image[:resized_image.shape[0], :resized_image.shape[1]] = resized_image
    return np.ascontiguousarray(padded_image, dtype=np.float32), ratio

def sigmoid(x): return 1 / (1 + np.exp(-x))
def softmax(x): return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=-1, keepdims=True)

def postprocess(outputs, img_size, ratio, nms_th, nms_score_th, max_width, max_height):
    strides = [8, 16, 32]
    hsizes = [img_size[0] // s for s in strides]
    wsizes = [img_size[1] // s for s in strides]
    grids, expanded_strides = [], []
    for h, w, s in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((*grid.shape[:2], 1), s))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    predictions = outputs[0]
    boxes = predictions[:, :4]
    scores = sigmoid(predictions[:, 4:5]) * softmax(predictions[:, 5:])
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
    valid = cls_scores > nms_score_th
    boxes, cls_scores, cls_inds = boxes_xyxy[valid], cls_scores[valid], cls_inds[valid]
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), cls_scores.tolist(), nms_score_th, nms_th)
    if len(indices) == 0: return [], [], []
    indices = indices.flatten()
    return boxes[indices], cls_scores[indices], cls_inds[indices]

def get_class(path):
    with open(path) as f:
        return [c.strip() for c in f.readlines()]

class_names = get_class(labels_file)
hsv_tuples = [(1.0 * x / len(class_names), 1., 1.) for x in range(len(class_names))]
colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)),
                  map(colorsys.hsv_to_rgb, hsv_tuples)))
random.seed(0)
random.shuffle(colors)
random.seed(None)

def draw_bbox(image, bboxes, classes):
    for bbox in bboxes:
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_id = int(bbox[5])
        color = colors[class_id]
        label = f"{classes[class_id]}: {score:.2f}"
        thickness = int(1.8 * (image.shape[0] + image.shape[1]) / 600)
        cv2.rectangle(image, tuple(coor[:2]), tuple(coor[2:]), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
        cv2.rectangle(image, (coor[0], coor[1] - th - 4), (coor[0] + tw, coor[1]), color, -1)
        cv2.putText(image, label, (coor[0], coor[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return image

# ***********************************************************************
# Run Inference
# ***********************************************************************
input_shape = (416, 416)
input_tensor = dpu.get_input_tensors()[0]
output_tensors = dpu.get_output_tensors()
input_data = [np.empty(tuple(input_tensor.dims), dtype=np.float32, order="C")]
output_data = [np.empty(tuple(t.dims), dtype=np.float32, order="C") for t in output_tensors]

cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! appsink", cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open the camera.")
    exit()

frame_count = 0
avg_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    img_data, ratio = preprocess(frame, input_shape)
    input_data[0][0,...] = img_data.reshape(input_tensor.dims[1:])

    start = time.time()
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)
    outputs = np.concatenate([out.reshape(1, -1, out.shape[-1]) for out in output_data], axis=1)
    bboxes, scores, class_ids = postprocess(outputs, input_shape, ratio, 0.45, 0.1, w, h)
    combined = [list(b) + [s, c] for b, s, c in zip(bboxes, scores, class_ids)]
    result = draw_bbox(frame, combined, class_names)
    
    cv2.imshow("YOLOX Detection", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print(f"FPS: {1 / (time.time() - start):.2f}")
    frame_count += 1
    if frame_count % 100 == 0:
        elapsed = time.time() - avg_start
        print(f"Average FPS over 100 frames: {100 / elapsed:.2f}")
        avg_start = time.time()

cap.release()
cv2.destroyAllWindows()

# ***********************************************************************
# Clean up
# ***********************************************************************
del overlay
del dpu

