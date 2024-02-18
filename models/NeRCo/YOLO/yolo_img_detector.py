import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


#Import YOLO Model
network = cv2.dnn.readNetFromDarknet('./yolo_files/yolov3.cfg', './yolo_files/yolov3.weights')
layers = network.getLayerNames()
yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']

# Path to the YOLO class names file
yolo_classes_file = './yolo_files/coco.names'  # Replace with the actual path to your YOLO class names file


# Load YOLO class names from the file
with open(yolo_classes_file, 'r') as file:
    yolo_classes = file.read().strip().split('\n')


#Importing in Images:

parser = argparse.ArgumentParser(description='Find location of images.')

parser.add_argument('--model', help='Path to folder containing input and output images', required=True)

# Parse the command-line arguments
args = parser.parse_args()

# Path to the folder containing images
#image_folder = './results/LOL/test_latest/images'  # Replace with the path to your image folder
image_folder = os.path.join(os.pardir, 'results', args.model, 'test_latest', 'images')

# List all image files in the folder
image_files = [f for f in os.listdir(image_folder)]

#Path to output folder or create folder if it doesn't exist
output_folder = os.path.join('./out_imgs/', args.model)
os.makedirs(os.path.join('out_imgs', args.model), exist_ok=True)


# Initialize the detection summary dictionaries
total_detections = {} #changed
detections_per_image = {} #changed


# Loop through each image in the folder and display it
for filename in image_files:
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)

    input_blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    network.setInput(input_blob)
    output = network.forward(yolo_layers)

    bounding_boxes = []
    confidences = []
    classes = []
    probability_minimum = 0.5
    threshold = 0.3
    h, w = image.shape[:2]

    for result in output:
      for detection in result:
          scores = detection[5:]
          class_current = np.argmax(scores)
          confidence_current = scores[class_current]
          if confidence_current > probability_minimum:
              box_current = detection[0:4] * np.array([w, h, w, h])
              x_center, y_center, box_width, box_height = box_current.astype('int')
              x_min = int(x_center - (box_width / 2))
              y_min = int(y_center - (box_height / 2))
              bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
              confidences.append(float(confidence_current))
              classes.append(class_current)

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
    coco_labels = 80
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')

 # Initialize the per-image count dictionary
    detections_per_image[filename] = {}

    if len(results) > 0:
      for i in results.flatten():
          class_id = classes[i]
          class_name = yolo_classes[class_id]
          total_detections[class_name] = total_detections.get(class_name, 0) + 1
          detections_per_image[filename][class_name] = detections_per_image[filename].get(class_name, 0) + 1

          x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
          box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
          colour_box = [int(j) for j in colours[classes[i]]]

          # Get the class name for the detected object
          class_name = yolo_classes[classes[i]]

          # Add class name and confidence to the text box
          text_box = f'{class_name} (conf: {confidences[i]:.4f})'

          cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height),
                        colour_box, 5)
          cv2.putText(image, text_box, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colour_box, 5)

          #Rename and save result images
          output_filename = os.path.splitext(filename)[0] + '_out.jpg'
          output_path = os.path.join(output_folder, output_filename)
          cv2.imwrite(output_path, image)


# After all images have been processed, generate the summary strings
total_detection_count = sum(total_detections.values())
total_breakdown_string = f'Total Number of Objects/Classes Detected: {total_detection_count}\n'
total_breakdown_string += 'Total Detection Breakdown\n'
for class_name, count in total_detections.items():
    total_breakdown_string += f'{class_name}: {count}\n'

per_image_breakdown_string = 'Per Image Breakdown\n'
for filename, counts in detections_per_image.items():
    per_image_breakdown_string += f'{filename} => '
    per_image_breakdown_string += ' | '.join([f'{class_name}: {count}' for class_name, count in counts.items()])
    per_image_breakdown_string += '\n'

# Combine all strings into one to save into a file
output_string = total_breakdown_string + '\n' + per_image_breakdown_string

# Define the output file path
output_file_path = os.path.join(output_folder, 'detection_summary.txt')

# Write the output string to the file
with open(output_file_path, 'w') as file:
    file.write(output_string)