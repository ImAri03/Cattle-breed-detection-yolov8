from ultralytics import YOLO
import os
import cv2

# Load your trained YOLOv8 model
model = YOLO("best.pt")

# Folders
input_folder = "input_images"
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# Loop through all images in input folder
for img_file in os.listdir(input_folder):
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, img_file)
        print(f"Processing {img_file}...")

        # Read image
        frame = cv2.imread(img_path)

        # Run YOLO inference
        results = model.predict(frame, device="cpu")  # use "cuda" if GPU is available

        # Annotate frame
        annotated_frame = results[0].plot()

        # Save annotated image
        save_path = os.path.join(output_folder, img_file)
        cv2.imwrite(save_path, annotated_frame)

print("All images processed! Check the 'output_images' folder.")
