import cv2
import numpy as np
import pyttsx3
import os
import time


##-----------MAC ONLY -------------
# Initialize a timestamp for the last time audio was played
last_audio_time = 0
audio_cooldown = 5  # Time in seconds before playing audio again
##-----------MAC ONLY -------------



# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("cfg/coco.data", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Analyze the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    current_time = time.time()

    for i in range(len(boxes)):
        class_id = class_ids[i]
        if i in indexes:
            x, y, w, h = boxes[i]

            center_x = x + w / 2
            # Define the boundaries for the center region (e.g., middle third of the frame)
            left_boundary = width / 3
            right_boundary = 2 * width / 3

            # Determine the side of the screen or if it's in the center
            if center_x < left_boundary:
                side = "left"
            elif center_x > right_boundary:
                side = "right"
            else:
                side = "center"

            color = (0, 255, 0)  # Use the color you want for the hue and outline

            # Draw a filled rectangle with transparency (color hue)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            alpha = 0.1  # Transparency factor.

            # Following line overlays transparent rectangle over the frame
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Draw the outline of the rectangle (the border)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # If the class_id corresponds to the ____ class
            if class_id == 0:
                label = "Person"

            elif class_id == 1:
                label = "Bicycle"

            elif class_id == 2:
                label = "Car"

            elif class_id == 10:
                label = "Traffic Light"

            elif class_id == 11:
                label = "Stop Sign"

            elif class_id == 13:
                label = "Bench"

            elif class_id == 56:
                label = "Chair"

            elif class_id == 16:
                label = "Dog"
            else:
                label = "Unknown" # This is the original label from the COCO dataset

            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            #------------MAC ONLY--------------
            if current_time - last_audio_time > audio_cooldown:
                # Prepare and play the text
                text = f"{label} on the {side} side"

                # Using macOS's say command for text-to-speech
                os.system(f'say "{text}"')

                # Update the last audio time
                last_audio_time = current_time
            #------------MAC ONLY--------------
            y = max(y, label_size[1])
            # Rest of your code for drawing boxes and labels...
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            print("Class ID out of range:", class_id)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break
        break


cap.release()
cv2.destroyAllWindows()
