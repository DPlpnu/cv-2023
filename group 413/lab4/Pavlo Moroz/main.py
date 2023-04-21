import cv2
import numpy as np

yolo = cv2.dnn.readNet("yolo.cnf", "yolo.weights")

image = cv2.imread("test2.webp")

blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)

yolo.setInput(blob)
outputs = yolo.forward(yolo.getUnconnectedOutLayersNames())



conf_threshold = 0.5
boxes = []
confidences = []
class_ids = []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x, center_y, width, height = (detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype('int')
            left, top = center_x - width//2, center_y - height//2
            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

nms_threshold = 0.4
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

with open(r'coco.names','r') as f:
    class_names=f.read().splitlines()

colors = np.random.uniform(0, 255, size=(len(boxes), 3))
for i in indices.flatten():
    left, top, width, height = boxes[i]
    final_accuracy=str(round(confidences[i],3))
    label = class_names[class_ids[i]] + " " + final_accuracy
    color = colors[i]
    cv2.rectangle(image, (left, top), (left + width, top + height), color, 2)
    cv2.putText(image, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


cv2.imshow("YOLO Output", image)
cv2.waitKey(0)
