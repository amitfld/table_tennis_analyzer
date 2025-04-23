import cv2
from ultralytics import YOLO
import sys
import os

def analyze_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # Load the model (make sure you have yolov8n-pose.pt downloaded)
    print("Loading model...")
    model = YOLO("yolo11n-pose.pt")

    # Run pose detection
    print("Running detection...")
    results = model(image)

    # Draw the results on the image
    print("Rendering results...")
    annotated_frame = results[0].plot()

    # Save the result to a file
    # Generate output path: inputname_pose.jpg
    base, ext = os.path.splitext(os.path.basename(image_path))
    output_path = f"{base}_pose{ext}"
    cv2.imwrite(output_path, annotated_frame)
    print(f"Annotated image saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python table_tennis_analyzer.py <image_path>")
    else:
        analyze_image(sys.argv[1])
