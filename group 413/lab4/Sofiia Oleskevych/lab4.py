categories = ['orthopedic crow', 'othopedic crown', 'orphopedic crown','crown']
right_name = 'orthopedic crown'

import os
import json

# Define the directory containing the JSON files
directory = "/Users/soles/Desktop/sorted_2"

# Iterate through each JSON file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        # Load the JSON file
        with open(os.path.join(directory, filename), "r") as f:
            data = json.load(f)

        # Modify the contents of the JSON file
        for i in data['shapes']:
            if i['label'] in categories:
                i['label'] = right_name

        # Rewrite the JSON file to disk
        with open(os.path.join(directory, filename), "w") as f:
            json.dump(data, f)

from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO

model = YOLO("yolov5s.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/Users/soles/Desktop/sorted_2/YOLODataset/dataset.yaml", epochs=10)  # train the model
metrics = model.val()
print(model.val().results_dict)

# test
test_path = '/Users/so/Desktop/sorted_2/YOLODataset/images/val/163.png'
test_image1 = cv2.imread(test_path)
results = model.predict([test_image1], save=True, line_thickness=1)
