# Hand Gesture Recognition System

A real-time hand gesture recognition system using MediaPipe and OpenCV that translates hand signs into text.

![Demo GIF](https://user-images.githubusercontent.com/37477845/102222442-c452cd00-3f26-11eb-93ec-c387c98231be.gif)

## Features

- Real-time hand tracking and gesture recognition
- Support for both static hand signs and dynamic finger gestures
- Built-in data collection mode for custom gesture training
- Pre-trained models for common hand signs
- FPS performance monitoring

## Requirements

- Python 3.10
- mediapipe 
- OpenCV 
- TensorFlow 
- scikit-learn 
- matplotlib 

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

### Command Line Options

- `--device`: Camera device ID (default: 0)
- `--width`: Camera capture width (default: 960)
- `--height`: Camera capture height (default: 540)
- `--min_detection_confidence`: Detection threshold (default: 0.5)
- `--min_tracking_confidence`: Tracking threshold (default: 0.5)

## Project Structure

```
├── app.py                         # Main application
├── model/
│   ├── keypoint_classifier/       # Hand sign recognition
│   │   ├── keypoint.csv          # Training data
│   │   ├── keypoint_classifier.tflite
│   │   └── keypoint_classifier_label.csv
│   └── point_history_classifier/  # Gesture recognition
│       ├── point_history.csv     # Training data
│       ├── point_history_classifier.tflite
│       └── point_history_classifier_label.csv
└── utils/
    └── cvfpscalc.py             # FPS calculator
```

## Training Custom Gestures

### Hand Signs (Static Gestures)

1. Press 'k' to enter keypoint logging mode
2. Press 'c' to collect data for curr and 'z' for next gesture classes
3. Train the model using `keypoint_classification.ipynb`

## Controls

- 'k': Enter keypoint logging mode
- 'c' or 'z': Record data for corresponding gesture class
- 'q': Quit application

## Model Architecture

### Hand Sign Classifier
- Input: 21 hand landmarks (x, y coordinates)
- Architecture: Simple MLP (Multi-Layer Perceptron)
- Output: Gesture classification

### Finger Gesture Classifier
- Input: Point history of finger movements
- Architecture: Choice of standard MLP or LSTM
- Output: Movement pattern classification

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Submit a pull request

## Acknowledgments

- [Debopriyo Das](https://github.com/dragon4926) - Project creator and maintainer
- MediaPipe team for the hand tracking solution

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
