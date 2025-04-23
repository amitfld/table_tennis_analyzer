import cv2
from ultralytics import YOLO
import sys
import os

# Constants
START_FRAME = 0
END_FRAME = 200  # Set to 0 to process the whole video

# Load the model
print("Loading model...")
model = YOLO("yolo11n-pose.pt")

def analyze_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Determine frame range
    start = START_FRAME
    end = END_FRAME if END_FRAME != 0 else total_frames

    # Set the video capture to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output_with_detections.mp4", fourcc, fps, (width, height))

    print(f"Processing frames {start} to {end}...")

    # Process each frame in the specified range
    for frame_idx in range(start, end):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_idx}")
            break

        # Run pose detection on the current frame
        results = model(frame)
        annotated_frame = results[0].plot()  # Draw poses and bounding boxes on the frame

        # Write the annotated frame to the output video
        out.write(annotated_frame)

    # Release resources
    cap.release()
    out.release()
    print("Processing complete. Output saved as output_with_detections.mp4")

if __name__ == "__main__":
    # Run the video analyzer on the specified input video
    analyze_video("input.mp4")
