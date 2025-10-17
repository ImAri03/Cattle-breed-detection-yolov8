from ultralytics import YOLO
import cv2

# Load your trained YOLOv8 model
model = YOLO("best.pt")

# Open default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model.predict(frame, device="cpu")  # Use "cuda" if GPU is supported

    # Annotate frame
    annotated_frame = results[0].plot()  # plots detections directly on the frame

    # Display live results
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
