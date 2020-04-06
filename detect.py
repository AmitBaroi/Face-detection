import numpy as np
import cv2
import os
from time import time

# Constants
CONFIDENCE = 0.5
IMG_DIR = "images"
MODEL_DIR = "model"

# Detect faces in images or web-cam video?
MODE = str(input("""Detect faces in,
- images
- webcam
>>> """)).lower()

# Load face detector model (Caffe model)
print("[INFO] Loading model...")
proto_txt_path = f"{MODEL_DIR}/deploy.prototxt.txt"
model_path = "model/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt=proto_txt_path, caffeModel=model_path)

# TODO: add functionality to save detection video to file


def detect_faces():
    # Making a blob (image pre-processing)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    # Set the blob as input of the model
    net.setInput(blob)

    start = time()
    # Perform a forward pass
    detections = net.forward()
    end = time()
    print(f"Detection took {round(end-start, 6)} seconds")

    # Iterate over each detection and draw bounding boxes
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # Filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > CONFIDENCE:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


# The main program
if __name__ == "__main__":
    # Detect faces in STATIC IMAGES
    if MODE in ['image', 'images']:
        for image_file in os.listdir(IMG_DIR):
            image_path = os.path.join(IMG_DIR, image_file)      # Full image path
            image = cv2.imread(image_path)                      # Load image as matrix
            (h, w) = image.shape[:2]                            # Get height and width
            print("[INFO] Processing: ", image_file)
            print(f"Image shape: {h}x{w}")

            # Face detection
            detect_faces()

            # Finally display the image
            cv2.imshow(image_file, image)
            cv2.waitKey(0)

    # Detect faces in WEBCAM VIDEO
    if MODE in ['webcam', 'web-cam']:
        cap = cv2.VideoCapture(0)
        while True:
            # Capture frame-by-frame
            ret, image = cap.read()
            (h, w) = image.shape[:2]  # Get height and width
            print(f"Image shape: {h}x{w}")

            # Face detection
            detect_faces()

            # Display the resulting frame
            cv2.imshow("Press 'q' to quit program", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
