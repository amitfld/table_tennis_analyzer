import cv2
from ultralytics import YOLO
import csv 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import scipy.ndimage.filters as filters

# Constants
START_FRAME = 1900
END_FRAME = 2250  # Set to 0 to process the whole video

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

    # Position storage for heatmap
    player1_coords = []
    player2_coords = []

    # Read frame 0 for background
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, background_frame = cap.read()
    if not ret:
        print("Failed to read first frame for background.")
        return
    # Set position back to START_FRAME for processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

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
            player1_coords.append(player_positions[0])
        else:
            row.extend(["", ""])

        # Add player 2 position (right-most)
        if len(player_positions) > 1:
            row.extend(player_positions[-1])
            player2_coords.append(player_positions[-1])
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

    print(f"\nDetected player1 in {len(player1_coords)} of {END_FRAME if END_FRAME != 0 else total_frames} frames.")
    print(f"Detected player2 in {len(player2_coords)} of {END_FRAME if END_FRAME != 0 else total_frames} frames.\n")

    # Create heatmap for each player separately and overlay both on the background frame
    def plot_dual_heatmap(coords1, coords2, background_image, title, output_filename):
        h, w, _ = background_image.shape
        data1 = np.zeros((h, w))
        data2 = np.zeros((h, w))

        for x, y in coords1:
            if 0 <= int(y) < h and 0 <= int(x) < w:
                data1[int(y)][int(x)] += 1

        for x, y in coords2:
            if 0 <= int(y) < h and 0 <= int(x) < w:
                data2[int(y)][int(x)] += 1

        # Apply Gaussian blur
        data1 = filters.gaussian_filter(data1, sigma=40)
        data2 = filters.gaussian_filter(data2, sigma=40)

        # Normalize independently
        norm1 = np.sqrt(data1)
        norm1 = norm1 / np.max(norm1) if np.max(norm1) > 0 else norm1
        norm2 = np.sqrt(data2)
        norm2 = norm2 / np.max(norm2) if np.max(norm2) > 0 else norm2

        # Create custom color map
        colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0.75), (0, 1, 0), (0.75, 1, 0),
                (1, 1, 0), (1, 0.8, 0), (1, 0.7, 0), (1, 0, 0)]
        cm = LinearSegmentedColormap.from_list('heatmap', colors)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))
        plt.imshow(norm1, cmap=cm, alpha=norm1, origin='upper')
        plt.imshow(norm2, cmap=cm, alpha=norm2, origin='upper')
        plt.colorbar(label='Relative Density')
        plt.title(title)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(output_filename)
        plt.close()

    # Plot combined normalized dual heatmap
    plot_dual_heatmap(player1_coords, player2_coords, background_frame,
                    "Combined Player Position Heatmap", f"player_position_heatmap.png")

    print("\nProcessing complete. Output saved as output_with_detections.mp4, player_positions.csv, and player_position_heatmap.png")
    
if __name__ == "__main__":
    # Run the video analyzer on the specified input video
    analyze_video("input.mp4")
