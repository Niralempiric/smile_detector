# Smile Detection Project

A real-time smile detection system using computer vision. This project provides two different approaches for detecting smiles: OpenCV Haar Cascades (simple and fast) and MediaPipe Face Mesh (more accurate).

## Features

- ✅ Real-time smile detection from webcam
- ✅ Static image smile detection
- ✅ Two implementation approaches (OpenCV and MediaPipe)
- ✅ Visual feedback with bounding boxes and status text
- ✅ Smile statistics tracking

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd smile_detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** If you encounter issues installing `dlib` or `mediapipe`, you may need to:
   - Install Visual C++ Build Tools (Windows)
   - Or use pre-built wheels: `pip install dlib-binary` (alternative)

## Usage

### Method 1: OpenCV Haar Cascades (Simple & Fast)

Run the basic OpenCV implementation:

```bash
python main.py
```

**Features:**
- Fast and lightweight
- Works well in good lighting conditions
- Uses pre-trained Haar cascades (included with OpenCV)

**Controls:**
- Press `q` to quit

**For static images:**
```bash
python main.py path/to/your/image.jpg
```

### Method 2: MediaPipe Face Mesh (More Accurate)

Run the MediaPipe implementation:

```bash
python mediapipe_smile_detection.py
```

**Features:**
- More accurate smile detection
- Uses facial landmarks for better precision
- Adjustable sensitivity threshold

**Controls:**
- Press `q` to quit
- Press `+` or `=` to increase sensitivity threshold
- Press `-` or `_` to decrease sensitivity threshold

## How It Works

### OpenCV Approach
- Uses Haar Cascade classifiers to detect faces
- Then applies a smile cascade to the lower half of detected faces
- Simple and fast, but may have false positives/negatives

### MediaPipe Approach
- Uses Face Mesh to detect facial landmarks
- Calculates smile ratio based on mouth width and height
- More accurate by analyzing actual facial geometry

## Project Structure

```
smile_detection/
├── requirements.txt              # Python dependencies
├── main.py                      # OpenCV-based smile detection
├── mediapipe_smile_detection.py # MediaPipe-based smile detection
└── README.md                    # This file
```

## Troubleshooting

### Webcam not working
- Make sure your webcam is connected and not being used by another application
- Try changing the camera index in the code: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`

### Poor detection accuracy
- Ensure good lighting conditions
- Face the camera directly
- For OpenCV: Adjust `scaleFactor` and `minNeighbors` parameters
- For MediaPipe: Adjust the `smile_threshold` value (use +/- keys)

### Installation issues
- Make sure you have Python 3.7 or higher
- On Windows, you may need to install Visual C++ Build Tools for some packages
- Try using a virtual environment:
  ```bash
  python -m venv venv
  venv\Scripts\activate  # Windows
  # or
  source venv/bin/activate  # Linux/Mac
  pip install -r requirements.txt
  ```

## Customization

### Adjusting OpenCV sensitivity:
Edit `main.py` and modify these parameters in `detect_smile()`:
- `scaleFactor=1.8` - Lower values = more sensitive
- `minNeighbors=20` - Lower values = more detections (but more false positives)

### Adjusting MediaPipe sensitivity:
- Use the `+` and `-` keys while running to adjust threshold in real-time
- Or edit `smile_threshold = 15.0` in `mediapipe_smile_detection.py`

## License

This project is open source and available for educational purposes.

## Future Enhancements

- [ ] Save detected smiles to files
- [ ] Add smile intensity scoring
- [ ] Support for multiple faces
- [ ] Video file processing
- [ ] Web interface
- [ ] Machine learning model training

