# Automated Cricket Video Analysis Using OCR and YOLOv8

## 📌 Project Overview

This project aims to automate the analysis of cricket match videos by identifying wicket-taking deliveries, extracting relevant video segments, and detecting the cricket pitch and ball using computer vision techniques.

Manually reviewing a complete cricket match takes considerable time. This system focuses on important wicket events so that coaches, analysts, and cricket enthusiasts can review relevant deliveries more efficiently.

The project combines **Optical Character Recognition (OCR), OpenCV, and YOLOv8 object detection** to process cricket broadcast videos and support post-match analysis.

## 🎯 Objectives

- Automatically read scorecard information from cricket videos.
- Identify potential wicket events by monitoring changes in the wicket count.
- Extract video clips associated with wicket-taking deliveries.
- Detect the cricket pitch and ball using YOLOv8 models.
- Analyze detected ball positions to estimate delivery trajectories.
- Support dismissal-zone analysis to identify recurring spatial patterns in wicket-taking deliveries.

## ⚙️ Project Workflow

The system follows these main stages:

1. **Input Cricket Video:** Load a cricket broadcast video for analysis.
2. **Scorecard Detection:** Locate the scorecard region in sampled video frames.
3. **OCR Processing:** Apply image preprocessing and OCR to read score and wicket information.
4. **Wicket Event Detection:** Compare successive valid score readings to identify possible wicket events.
5. **Clip Extraction:** Extract relevant video segments around detected wicket events.
6. **Pitch Detection:** Use a trained YOLOv8 model to locate the cricket pitch.
7. **Ball Detection:** Use a trained YOLOv8 model to detect the cricket ball.
8. **Trajectory Analysis:** Arrange detected ball positions in temporal order to estimate the delivery path.
9. **Dismissal-Zone Analysis:** Visualize spatial patterns from wicket-taking deliveries to support tactical review.

## 🛠️ Technologies Used

- **Python** – Main programming language
- **OpenCV** – Video processing, frame extraction, cropping, and image preprocessing
- **EasyOCR** – Text recognition from scorecard images
- **YOLOv8 (Ultralytics)** – Cricket pitch and ball detection
- **NumPy** – Numerical operations and coordinate processing
- **Pandas** – Storing and analyzing OCR and event results
- **Google Colab** – Cloud-based development and execution
- **Jupyter Notebook** – Organizing the analysis pipeline

## 🧠 Main Components

### 1. OCR-Based Wicket Detection

The system samples frames from the cricket video and extracts the scorecard region. Image preprocessing techniques, such as grayscale conversion, contrast enhancement, and morphological operations, help improve text readability.

OCR reads the scorecard information, and successive valid readings are compared to identify possible wicket-count increases.

### 2. Wicket Delivery Clip Extraction

When a potential wicket event is identified, the system extracts a relevant video segment for further analysis.

This reduces the need to process and review every delivery manually. The accuracy of event detection depends on scorecard visibility, correct OCR readings, and the event-handling logic.

### 3. Cricket Pitch Detection

A trained YOLOv8 model detects the pitch region in video frames. The pitch provides a spatial reference for filtering ball detections and studying delivery movement.

### 4. Cricket Ball Detection

A separate trained YOLOv8 model detects the cricket ball. Ball detection is challenging because the ball is small, moves quickly, and can be affected by motion blur, occlusion, and camera movement.

### 5. Ball Trajectory and Dismissal-Zone Analysis

Detected ball positions can be arranged in temporal order to estimate the delivery trajectory. Combining trajectories from multiple wicket-taking deliveries can help visualize recurring dismissal-associated regions.

These visualizations are intended to support tactical review; they do not independently establish why a batter was dismissed.

## 📊 Reported Model Performance

The following values are reported in the reference research paper, *Automated Wicket-Taking Delivery Segmentation and Trajectory-Based Dismissal-Zone Analysis in Cricket Videos Using OCR-Guided YOLOv8*.

| Model | Precision | Recall | mAP@50 |
|---|---:|---:|---:|
| Pitch Detection | 0.999 | 1.000 | 0.995 |
| Ball Detection (Transfer Learning) | 0.968 | 0.978 | 0.9918 |

These are the paper's reported experimental results, not a guarantee of performance for every model or video used in this project. Evaluate your own trained weights on the appropriate validation or test dataset before reporting your project's final metrics.

## 🚀 How to Run the Project

### Requirements

- A Google account to use Google Colab
- A cricket match video that you are permitted to use
- The required Python libraries
- Trained YOLOv8 model weights for pitch and ball detection
- Access to the appropriate datasets if retraining or evaluation is required

### Steps

1. Clone or download this repository.
2. Open the required `.ipynb` notebook in Google Colab.
3. Install the dependencies specified in `requirements.txt` or the notebook.
4. Upload or mount the input cricket video in Colab.
5. Upload the trained pitch and ball model weights when prompted.
6. Run the notebook cells in order.
7. Review the OCR output, detected wicket events, extracted clips, and object-detection results.
8. If trajectory and dismissal-zone visualization is implemented, review those outputs as well.

**Important:** The notebooks may contain Google Colab-specific code and may not run unchanged in a local Python environment.

## 📁 Datasets and Model Weights

The project uses annotated cricket images to train or evaluate object-detection models.

- Pitch detection dataset: [Cricket Pitch Detection Dataset](https://universe.roboflow.com/cricket-pitch/cricket-pitch-t9j9g)
- Ball detection dataset: [Cricket Ball Detection Dataset](https://universe.roboflow.com/daniyalworkpace/cricket-ball-detection-8uv1o)

Check each dataset's current licence and usage conditions before downloading or redistributing it.

Large datasets and trained model files, such as `.pt` weights, are not included in this repository. Place the required weights in the expected location or upload them when running the notebook.

## ⚠️ Limitations

- OCR may fail when the scorecard is missing, blurry, obstructed, or displayed in an unfamiliar layout.
- Incorrect score readings can produce missed or false wicket events.
- Ball detection may be affected by fast movement, small object size, occlusion, and camera motion.
- Camera panning, tilting, and zooming can complicate trajectory estimation.
- Estimated trajectories from broadcast footage may not represent the ball's exact three-dimensional path.
- Dismissal-zone visualizations provide analytical clues and should be interpreted alongside match context.

## 🔮 Future Improvements

- Improve OCR accuracy across different scorecard layouts.
- Handle innings changes and score resets more reliably.
- Improve ball tracking across frames when detections are missing.
- Apply camera calibration or perspective correction for better spatial analysis.
- Add interactive trajectory and dismissal-zone heatmap visualizations.
- Evaluate the complete pipeline on multiple matches using clearly defined metrics.
- Develop a user-friendly interface for uploading videos and viewing analysis results.

## 👥 Applications

This project may support:

- Cricket coaching and tactical analysis
- Post-match performance review
- Identification of recurring wicket-taking patterns
- Video indexing and event-based clip generation
- Cricket analytics and computer-vision research




