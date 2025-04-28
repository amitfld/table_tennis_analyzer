import cv2
from ultralytics import YOLO
import csv 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import scipy.ndimage.filters as filters
import easyocr

# Constants
START_FRAME = 0
END_FRAME = 7050  # Set to 0 to process the whole video

# Load the model
print("Loading model...")
model = YOLO("yolo11n-pose.pt")

# Initialize EasyOCR once
reader = easyocr.Reader(['en', 'ru'])

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

    # Scores storage for score chart
    player1_scores = []
    player2_scores = []
    names_detected = False

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

        # Check if frame is valid
        if not is_valid_frame(boxes, width, height):
            print(f"Invalid frame number: {frame_idx}")
            continue

        if frame_idx % 50 == 0:
            if not names_detected:
                player1_name, player2_name = detect_player_names(frame)
                if player1_name is not None and player2_name is not None:
                    names_detected = True
            calculated_score_1, calculated_score_2 = detect_scores(frame)
            
            # Add to lists (handle None safely)
            if calculated_score_1 is not None and calculated_score_2 is not None:
                player1_scores.append(calculated_score_1)
                player2_scores.append(calculated_score_2)
            else:
                # If detection failed, repeat last known score (optional)
                if player1_scores:
                    player1_scores.append(player1_scores[-1])
                if player2_scores:
                    player2_scores.append(player2_scores[-1])
                    

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

    print(f"\nDetected player1 in {len(player1_coords)} of {END_FRAME - START_FRAME if END_FRAME != 0 else total_frames} frames.")
    print(f"Detected player2 in {len(player2_coords)} of {END_FRAME - START_FRAME if END_FRAME != 0 else total_frames} frames.\n")

    # How many points did we collect?
    num_points = len(player1_scores)

    # Create X-axis as seconds
    seconds_x = list(range(num_points))  # 0,1,2,3,4...

    # Then plot
    plt.figure(figsize=(10, 6))
    plt.plot(seconds_x, player1_scores, label=player1_name, marker='o')
    plt.plot(seconds_x, player2_scores, label=player2_name, marker='s')
    plt.xlabel('Seconds')
    plt.ylabel('Calculated Score')
    plt.title('Players\' Scores Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig("score_chart.png")
    plt.show()

    print("\nScore chart saved as score_chart.png")


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

def is_valid_frame(boxes, frame_width, frame_height):
    """
    Determines whether a video frame is valid for analysis based on player detection.

    A frame is considered valid if:
    - It contains between 3 to 5 detected people (typically 2 players and a referee).
    - No detected person is overly large in height (to filter out close-up views).
    - No detected person appears near the bottom of the screen and is large (to avoid rear-view replays).
    - The leftmost and rightmost people are far enough apart (to ensure a proper side-view).

    Parameters:
        boxes (Boxes): The detection result containing bounding boxes from the YOLO model.
        frame_width (int): Width of the video frame.
        frame_height (int): Height of the video frame.

    Returns:
        bool: True if the frame passes all validity checks, False otherwise.
    """

    if boxes is None or not (3 <= len(boxes) <= 5):
        return False

    people = []

    for box in boxes.xyxy:
        x1, y1, x2, y2 = box[:4].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        box_height = y2 - y1

        # Skip if someone is too tall (close-up)
        if box_height / frame_height > 0.5:
            print("Person too big")
            return False

        # Skip if someone is close to the bottom and also big (likely rear-view)
        if y2 > 0.85 * frame_height and box_height / frame_height > 0.25:
            print(f"Too clost to bottom and too big: {box_height / frame_height}")
            return False

        people.append((cx, cy))

    # Sort people by x (left to right)
    people.sort(key=lambda p: p[0])
    leftmost = people[0]
    rightmost = people[-1]

    # Skip if players are too close together (replay, zoom-in)
    horizontal_distance = abs(leftmost[0] - rightmost[0])
    if horizontal_distance < 0.25 * frame_width:
        print("Players too close together")
        return False

    return True

def detect_player_names(frame):
    h, w, _ = frame.shape
    top = int(0.83 * h)
    bottom = int(0.965 * h)
    left = int(0.081 * w)
    right = int(0.3 * w)

    name_region = frame[top:bottom, left:right]
    name_region = cv2.resize(name_region, (2 * (right - left), 2 * (bottom - top)), interpolation=cv2.INTER_LINEAR)

    results = reader.readtext(name_region, low_text=0.3)

    player_names = []
    for bbox, text, confidence in results:
        clean_text = text.strip()
        if clean_text.isalpha() or ' ' in clean_text:  # crude check for names
            player_names.append(clean_text)

    if len(player_names) >= 2:
        player1_name = player_names[0]
        player2_name = player_names[1]
        print(f"Detected Player 1: {player1_name}")
        print(f"Detected Player 2: {player2_name}")
        return player1_name, player2_name
    else:
        print("Failed to detect both player names.")
        return None, None

def detect_scores(frame):
    h, w, _ = frame.shape
    top = int(0.83 * h)
    bottom = int(0.965 * h)
    left = int(0.23 * w)
    right = int(0.29 * w)

    score_region = frame[top:bottom, left:right]
    score_region = cv2.resize(score_region, (2 * (right - left), 2 * (bottom - top)), interpolation=cv2.INTER_LINEAR)

    results = reader.readtext(score_region, allowlist ='0123456789', low_text=0.3, mag_ratio=2)

    numbers = []
    for bbox, text, confidence in results:
        text = text.strip()
        if text.isdigit():
            numbers.append(int(text))
    
    if len(numbers) not in [2, 4]:
        results = reader.readtext(score_region, allowlist ='0123456789', low_text=0.3, mag_ratio=4)

        numbers = []
        for bbox, text, confidence in results:
            text = text.strip()
            if text.isdigit():
                numbers.append(int(text))

    if len(numbers) == 2:
        games_won_1 = numbers[0]
        games_won_2 = numbers[1]
        calculated_score_1 = games_won_1 * 10
        calculated_score_2 = games_won_2 * 10
        return calculated_score_1, calculated_score_2
    elif len(numbers) == 4:
        games_won_1 = numbers[0]
        points_1 = numbers[1]
        games_won_2 = numbers[2]
        points_2 = numbers[3]
        print(f"Player 1 won {games_won_1} games and has {points_1} points")
        print(f"Player 2 won {games_won_2} games and has {points_2} points")
        calculated_score_1 = games_won_1 * 10 + points_1
        calculated_score_2 = games_won_2 * 10 + points_2
        return calculated_score_1, calculated_score_2
    else:
        print("Failed to detect two scores.")
        return None, None

if __name__ == "__main__":
    # Run the video analyzer on the specified input video
    analyze_video("input.mp4")
