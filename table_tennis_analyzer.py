import cv2
from ultralytics import YOLO
import sys
import csv 

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

    # Prepare CSV for writing player positions
    csv_file = open("player_positions.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "player1_position_x", "player1_position_y", "player2_position_x", "player2_position_y"])

    print(f"Processing frames {start} to {end}...")

    # Process each frame in the specified range
    for frame_idx in range(start, end):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_idx}")
            break

        # Run pose detection on the current frame
        results = model(frame)
        boxes = results[0].boxes  # Draw poses and bounding boxes on the frame
        player_positions = []

        # Extract center points of all detected persons
        if boxes is not None:
            for box in boxes.xyxy:  # (x1, y1, x2, y2)
                x1, y1, x2, y2 = box[:4].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                player_positions.append((cx, cy))

        # Sort players by x-position to identify player1 (left) and player2 (right)
        player_positions.sort(key=lambda p: p[0])

        # Initialize CSV row
        row = [frame_idx + 1]

        # Add player 1 position (left-most)
        if len(player_positions) > 0:
            row.extend(player_positions[0])
        else:
            row.extend(["", ""])

        # Add player 2 position (right-most)
        if len(player_positions) > 1:
            row.extend(player_positions[-1])
        else:
            row.extend(["", ""])

        # Write to CSV
        csv_writer.writerow(row)

        # Annotate and write video frame
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    # Release resources
    cap.release()
    out.release()
    csv_file.close()
    print("Processing complete. Output saved as output_with_detections.mp4 and player_positions.csv")

if __name__ == "__main__":
    # Run the video analyzer on the specified input video
    analyze_video("input.mp4")
