import cv2
from ultralytics import YOLO
import csv 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import scipy.ndimage.filters as filters
import easyocr

# Constants
START_FRAME = 59300
END_FRAME = 61500  # Set to 0 to process the whole video

# Load the model
print("\nLoading model...")
model = YOLO("yolo11n-pose.pt")

# Initialize EasyOCR once
print("\nInitialize EasyOCR...")
reader = easyocr.Reader(['en', 'ru'])

def analyze_video(video_path):
    # Open the video file
    print("\n\nOpening video file")
    cap = cv2.VideoCapture(video_path)
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"\nFrames Per Second = {FPS}")

    if not cap.isOpened():
        print(f"\nError opening video file: {video_path}")
        return
    
    # Get video properties
    print("\nGetting video properties")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Determine frame range
    start = START_FRAME
    end = END_FRAME if END_FRAME != 0 else total_frames
    print(f"\nframe range is: {start} to {end}")
    print(f"\nTotal of: {end - start} frames")

    # Set the video capture to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output_with_detections.mp4", fourcc, fps, (width, height))

    # Prepare CSV for writing player positions
    print("\n\nPrepare CSV for writing player positions")
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
    print("\n\nCapturing first frame for backgroung")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, background_frame = cap.read()
    if not ret:
        print("\nFailed to read first frame for background.")
        return
    # Set position back to START_FRAME for processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    print(f"\n\nStart processing frames {start} to {end}...")

    # Process each frame in the specified range
    for frame_idx in range(start, end):
        ret, frame = cap.read()
        if not ret:
            print(f"\nFailed to read frame {frame_idx}")
            break

        # Run pose detection on the current frame
        results = model(frame)
        boxes = results[0].boxes  # Draw poses and bounding boxes on the frame
        player_positions = []

        # Check if frame is valid
        if not is_valid_frame(boxes, width, height):
            print(f"\nInvalid frame number: {frame_idx}")
            continue

        if frame_idx % FPS == 0:
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
    plt.plot(seconds_x, player1_scores, label=player1_name) # Removed , marker='o'
    plt.plot(seconds_x, player2_scores, label=player2_name) # Removed , marker='s'
    plt.xlabel('Seconds')
    plt.ylabel('Calculated Score')
    plt.title('Players\' Scores Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig("score_chart.png")

    print("\nScore chart saved as score_chart.png\n")


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
    
    generate_animated_heatmap('player_positions.csv', 'input.mp4', output_path='animated_heatmap.mp4')

    print("\n✅ Processing complete. " \
    "Output files saved as " \
    "output_with_detections.mp4, player_positions.csv, player_position_heatmap.png, score_chart.png, and animated_heatmap.mp4")

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
        if games_won_1 > 3 or games_won_2 > 3:
            print("Failed to detect two scores.")
            return None, None
        calculated_score_1 = games_won_1 * 10
        calculated_score_2 = games_won_2 * 10
        return calculated_score_1, calculated_score_2
    elif len(numbers) == 4:
        games_won_1 = numbers[0]
        points_1 = numbers[1]
        games_won_2 = numbers[2]
        points_2 = numbers[3]
        if games_won_1 > 3 or games_won_2 > 3 or points_1 > 10 or points_2 > 10:
            print("Failed to detect two scores.")
            return None, None
        print(f"Player 1 won {games_won_1} games and has {points_1} points")
        print(f"Player 2 won {games_won_2} games and has {points_2} points")
        calculated_score_1 = games_won_1 * 10 + points_1
        calculated_score_2 = games_won_2 * 10 + points_2
        return calculated_score_1, calculated_score_2
    else:
        print("Failed to detect two scores.")
        return None, None

def generate_animated_heatmap(csv_path, video_path, output_path="animated_heatmap.mp4"):
    # Settings
    SIGMA = 35
    UPDATE_EVERY_N_FRAMES = 25
    ALPHA = 0.65
    THRESHOLD = 0.2

    # Load first frame
    print("\nLoading first frame for heatmap...")
    video = cv2.VideoCapture(video_path)
    ret, first_frame = video.read()
    video.release()

    if not ret:
        raise ValueError("\nFailed to load first frame from video!")

    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = first_frame.shape

    # Read player positions
    player1_coords = []
    player2_coords = []

    with open(csv_path, 'r') as csvfile:
        print("Reading player positions...")
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['player1_position_x'] and row['player1_position_y']:
                x1 = float(row['player1_position_x'])
                y1 = float(row['player1_position_y'])
                player1_coords.append((x1, y1))
            if row['player2_position_x'] and row['player2_position_y']:
                x2 = float(row['player2_position_x'])
                y2 = float(row['player2_position_y'])
                player2_coords.append((x2, y2))

    total_frames = min(len(player1_coords), len(player2_coords))
    print(f"\nTotal frames for heatmap: {total_frames}\n")

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 50, (frame_width, frame_height))

    # Initialize heatmaps
    heatmap1 = np.zeros((frame_height, frame_width))
    heatmap2 = np.zeros((frame_height, frame_width))

    # Process frames manually
    frame_idx = 0
    while frame_idx < total_frames:
        frame = first_frame_rgb.copy()

        # Accumulate points for the next 25 frames
        end_idx = min(frame_idx + UPDATE_EVERY_N_FRAMES, total_frames)
        for i in range(frame_idx, end_idx):
            x1, y1 = player1_coords[i]
            x2, y2 = player2_coords[i]

            if 0 <= int(y1) < frame_height and 0 <= int(x1) < frame_width:
                heatmap1[int(y1), int(x1)] += 1
            if 0 <= int(y2) < frame_height and 0 <= int(x2) < frame_width:
                heatmap2[int(y2), int(x2)] += 1

        # Blur and combine
        blurred1 = filters.gaussian_filter(heatmap1, sigma=SIGMA)
        blurred2 = filters.gaussian_filter(heatmap2, sigma=SIGMA)

        # Square root and normalize separately
        blurred1 = np.sqrt(blurred1)
        blurred2 = np.sqrt(blurred2)
        blurred1 = blurred1 / np.max(blurred1) if np.max(blurred1) > 0 else blurred1
        blurred2 = blurred2 / np.max(blurred2) if np.max(blurred2) > 0 else blurred2

        combined = blurred1 + blurred2
        combined = np.clip(combined, 0.0, 1.0)

        # Remove weak heat
        combined[combined < THRESHOLD] = 0

        # Color
        colors = [
            (0, 0, 1),
            (0, 1, 1),
            (0, 1, 0.75),
            (0, 1, 0),
            (0.75, 1, 0),
            (1, 1, 0),
            (1, 0.8, 0),
            (1, 0.7, 0),
            (1, 0, 0)
        ]
        custom_cmap = LinearSegmentedColormap.from_list('custom_heat', colors, N=256)
        colored_heatmap = custom_cmap(combined)
        colored_heatmap_rgb = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)

        # Mask blending
        mask = (combined > 0.05).astype(np.float32)
        mask = mask[:, :, np.newaxis]

        blended = frame * (1 - mask * ALPHA) + colored_heatmap_rgb * (mask * ALPHA)
        blended = blended.astype(np.uint8)

        final_frame = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        # Write 25 frames with the same heatmap
        # leave VideoWriter at 50 FPS
        DUPLICATES = UPDATE_EVERY_N_FRAMES // 4   # 25/4 ≈ 6 → ~×4 speed
        for _ in range(DUPLICATES):
            out.write(final_frame)

        frame_idx += UPDATE_EVERY_N_FRAMES

    out.release()


if __name__ == "__main__":
    # Run the video analyzer on the specified input video
    analyze_video("input.mp4")
