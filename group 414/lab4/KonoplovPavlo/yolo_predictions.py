#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader


INPUT_WH_YOLO = 640
CONF_VALUE = 0.25
PROB_VALUE = 0.15


class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        with open(data_yaml, mode='r') as file:
            data_yaml = yaml.load(file, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.classes_number = data_yaml['number_of_classes']

        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self, image):
        row, col, d = image.shape
        max_res = max(row, col)
        input_image = np.zeros((max_res, max_res, 3), dtype=np.uint8)
        input_image[:row,:col] = image

        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()

        detections = preds[0]
        boxes = list([])
        confidences = list([])
        classes = list([])

        image_width, image_height = input_image.shape[:2]
        x_factor = image_width / INPUT_WH_YOLO
        y_factor = image_height / INPUT_WH_YOLO

        for row in detections:
            confidence = row[4]
            if confidence > CONF_VALUE:
                class_score = row[5:].max()
                class_id = row[5:].argmax()

                if class_score > PROB_VALUE:
                    cx, cy, w, h = row[:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])

                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)


        # NMS
        index = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.15).flatten()


        # Draw the Bounding
        for i in index:
            x, y, w, h = boxes[i]
            bb_conf = int(confidences[i] * 100)
            classes_id = classes[i]
            class_name = self.labels[classes_id]
            colors = self.generate_colors(classes_id)

            text = f'{class_name}: {bb_conf}%'

            cv2.rectangle(image, (x, y), (x+w, y+h), colors, 2)
            cv2.rectangle(image, (x, y-30), (x+w, y), colors, -1)

            cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0), 1)
            
        return image
    
    def generate_colors(self, ID):
        np.random.seed(1)
        colors = np.random.randint(100, 255, size=(self.classes_number, 3)).tolist()
        return tuple(colors[ID])
        