import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("yolov8x.pt")  # Replace with your custom model (e.g., "best.pt")

# Open the video file (or use 0 for webcam)
# video_path = "input_video.mp4"
cap = cv2.VideoCapture(0)
print("hello")
# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter to save the output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break when the video ends

    # Run YOLO on the frame
    results = model(frame)

    # Draw the detected objects on the frame
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = model.names[cls]  # Get class name

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow("YOLO Video Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
