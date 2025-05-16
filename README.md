# Real-Time Computer Vision Detection

A comprehensive computer vision application that performs real-time detection and analysis of faces, pose, hands, and motion using webcam input. The project combines multiple computer vision capabilities to create an interactive and informative experience.

## Features

- **Face Mesh Detection**: Displays 468 facial landmarks in real-time
- **Distance Estimation**: Approximates the distance of face from camera
- **Eye Blink Detection**: Detects and indicates when the user blinks
- **Emotion Recognition**: Analyzes and displays the dominant emotion
- **Pose Detection**: Tracks body pose and calculates joint angles
- **Hand Gesture Recognition**: Detects hand landmarks and recognizes gestures (e.g., index finger up)
- **Motion Detection**: Identifies and alerts when motion is detected
- **Recording Capabilities**: 
  - Video recording with 'r' key toggle
  - Screenshot capture with 's' key
- **Performance Metrics**: Real-time FPS display

## Requirements

```
opencv-python
mediapipe
numpy
deepface
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RealTimeDectorComputerVision.git
cd RealTimeDectorComputerVision
```

2. Install the required packages:
```bash
pip install opencv-python mediapipe numpy deepface
```

## Usage

Run the main script:
```bash
python RealTimeComputerVisionDetection.py
```

### Controls:
- Press 'q' to quit the application
- Press 'r' to toggle recording
- Press 's' to take a screenshot

## Output

- Video output is saved as 'output.avi'
- Screenshots are saved as 'screenshot_[number].jpg'
- Real-time display shows:
  - FPS counter
  - Distance estimation
  - Emotion detection
  - Blink detection
  - Motion detection
  - Recording status

## Technical Details

The application uses several key computer vision libraries:
- **MediaPipe**: For face mesh, pose, and hand landmark detection
- **OpenCV**: For video capture and image processing
- **DeepFace**: For emotion recognition
- **NumPy**: For numerical computations

## Contributing

Feel free to fork the project and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/) 