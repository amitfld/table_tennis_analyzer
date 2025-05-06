# 🏓 Table Tennis Analyzer

This project is a **computer vision tool** designed to **analyze table tennis matches**. It tracks player positions, scores, and movement patterns from video footage, providing rich **visual insights** like annotated videos, heatmaps, and score charts.

> ⚠️ **Disclaimer:**  
> This tool is for **educational and research purposes only**. It was developed as part of a university course on AI applications and computer vision, and is not affiliated with or endorsed by any professional sports body.

---

## 🚀 Features

- **Pose & Player Detection:**  
  Detects players using the YOLO pose model (`yolo11n-pose.pt`), draws bounding boxes and skeleton joints in each video frame.

- **Video Processing:**  
  Annotates each frame of a video, outputs a new video (`output_with_detections.mp4`) showing live detections.

- **Player Position Tracking:**  
  Exports player center positions frame-by-frame to a CSV (`player_positions.csv`) with columns:
  ```
  frame, player1_position_x, player1_position_y, player2_position_x, player2_position_y
  ```
  Players are identified by **left-most (Player 1)** and **right-most (Player 2)** logic.

- **Heatmaps:**  
  Creates a **combined player heatmap** (`player_position_heatmap.png`) showing player movement density across the match.  
  Includes an **animated heatmap** (`animated_heatmap.mp4`) that **updates dynamically over time**.

- **Score Detection:**  
  Uses **EasyOCR** to read in-game scores directly from the video frames, and generates a **score chart** (`score_chart.png`) showing both players’ calculated scores over time.

- **Invalid Frame Filtering:**  
  Implements custom `is_valid_frame` logic to **skip invalid frames** (e.g., replays, zoom-ins, non-side views) to maintain clean data.

- **Automatic Name & Score Extraction:**  
  Detects player names and scores periodically and logs them to provide context in the visualizations.

- **Final Outputs:**  
  After running the analysis, you will get the following:
  - `output_with_detections.mp4`
  - `player_positions.csv`
  - `player_position_heatmap.png`
  - `animated_heatmap.mp4`
  - `score_chart.png`

---

## 🛠 Technologies & Libraries

| Tool/Library             | Purpose                                                      |
|--------------------------|--------------------------------------------------------------|
| **YOLO (ultralytics)**   | Pose estimation & player detection                           |
| **OpenCV**               | Video frame handling & image processing                      |
| **EasyOCR**              | Extracts text (player names and scores) from frames          |
| **Matplotlib**           | Heatmaps and score chart generation                          |
| **SciPy (Gaussian filter)** | Smooths heatmap data for better visuals                   |
| **NumPy**                | Array & math operations                                      |
| **Custom Utils**         | Invalid frame detection, score calculation, heatmap animation|

See [`requirements.txt`](requirements.txt) for the full list of dependencies.

---

## 📈 How It Works

1️⃣ **Load the Video**  
Reads your `input.mp4` file and iterates through each frame (configurable via `START_FRAME` and `END_FRAME`).

2️⃣ **Player Detection**  
For each valid frame:
- Detects players and draws bounding boxes & pose skeletons.
- Identifies **Player 1** and **Player 2** based on horizontal position.
- Saves their center coordinates to CSV.

3️⃣ **Score & Name Detection**  
- Every second (based on FPS), the tool uses OCR to:
  - Extract player names.
  - Read scores and calculate a **calculatedScore = gamesWon * 10 + currentGamePoints**.

4️⃣ **Visual Outputs**
- **Annotated Video:** Highlights players and poses.
- **CSV Export:** Logs all player positions.
- **Score Chart:** Plots scores over time.
- **Heatmaps:** 
  - Static heatmap showing cumulative player movement.
  - Animated heatmap that evolves over time.

5️⃣ **Post-Processing**
- All files are saved in the working directory, ready for inspection and reporting.

---

## 🖼 Example Outputs

| Output File                         | Description                                                         |
|-------------------------------------|---------------------------------------------------------------------|
| `output_with_detections.mp4`        | Video with bounding boxes + poses drawn on players                  |
| `player_positions.csv`              | Player center positions for each frame                              |
| `player_position_heatmap.png`       | Cumulative heatmap of player movement                               |
| `animated_heatmap.mp4`              | Animated heatmap evolving over time                                 |
| `score_chart.png`                   | Line chart of player scores over time                               |

---

## 🖥️ How To Run

Follow these steps to set up and run the Table Tennis Analyzer:

1️⃣ **Clone the Repository**

```bash
git clone https://github.com/amitfld/table-tennis-analyzer.git
cd table-tennis-analyzer
```

2️⃣ **Place Your Video**

- Add your input video file to the project directory and **rename it to:**

```
input.mp4
```

3️⃣ **Set Up a Virtual Environment**

It’s recommended to use a virtual environment to manage dependencies cleanly:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4️⃣ **Install Required Packages**

Make sure you have `pip` upgraded, then install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5️⃣ **Adjust Video Frame Range (Optional)**

- Open `table_tennis_analyzer.py`.
- Modify these two variables at the top of the file to specify which part of the video you want to analyze:

```python
START_FRAME = 0      # Change to your desired start frame
END_FRAME = 0        # Change to your desired end frame (0 = full video)
```

For example, to analyze frames 100–500:

```python
START_FRAME = 100
END_FRAME = 500
```

6️⃣ **Run the Analyzer**

Start the analysis with:

```bash
python table_tennis_analyzer.py
```

---

✔️ **The program will:**

- Process the specified video frames.
- Detect and annotate players with bounding boxes and poses.
- Save player positions to `player_positions.csv`.
- Create:
  - `output_with_detections.mp4`
  - `player_position_heatmap.png`
  - `animated_heatmap.mp4`
  - `score_chart.png`

All files will be saved in the project directory.

---

## 🙌 Acknowledgments

This project is part of the From Idea to Reality Using AI course. Thanks to the instructor for guidance

---

## 👤 Author

Made with ❤️ for the **"Idea to Reality" 2025 course.**

GitHub: [@amitfld](https://github.com/amitfld)

LinkdIn: [Amit Feldman](https://www.linkedin.com/in/amit-fld/)

