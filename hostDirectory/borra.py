import cv2
from ultralytics import YOLO

import time
interval=1
base_time = time.time()

# Load the YOLOv8 model
model = YOLO('yolov8l.pt')

# Open the video file
video_path = "soccer.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    if(interval<time.time() - base_time):
        base_time=time.time()

        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            #results = model(frame)
            #results = model.track(source=frame) 
            results = model.track(source=frame, persist=True, tracker='custom_tracker.yaml') 

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
