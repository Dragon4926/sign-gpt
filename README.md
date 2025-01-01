# Sign GPT

## Overview
This project envisions AI as a companion for sign language users, especially non-verbal individuals. It interprets gestures, processes them via an AI model, and responds in text or audio in the user's native language. By bridging language barriers, it promotes inclusivity, independence, and real-time communication.

## Features
- **Real-Time Gesture Recognition**: Detect and classify hand gestures in real-time.
- **Customizable Data Collection**: Collect data for different hand gestures and log them into CSV files.
- **Visual Feedback**: Display bounding boxes, landmarks, and gesture classifications on the screen.
- **Support for Static and Dynamic Modes**: Enable static image mode or track dynamic hand movements.
- **Language Output**: Generate responses in text or audio in the user's native language.

## Prerequisites

1. **Python**: Ensure Python 3.7 or higher is installed.
2. **Libraries**:
   - `opencv-python`
   - `numpy`
   - `mediapipe`
3. **Model Files**:
   - KeyPoint Classifier and Point History Classifier models (`model/keypoint_classifier` and `model/point_history_classifier`).
   - Corresponding label files in CSV format.

You can install the required Python libraries using the command:

```bash
pip install opencv-python mediapipe numpy
```

## Usage

### Command-Line Arguments
The script supports the following command-line arguments:

- `--device`: Camera device ID (default: 2).
- `--width`: Width of the video capture (default: 960).
- `--height`: Height of the video capture (default: 540).
- `--use_static_image_mode`: Enable static image mode for debugging.
- `--min_detection_confidence`: Minimum confidence for detection (default: 0.7).
- `--min_tracking_confidence`: Minimum confidence for tracking (default: 0.5).

### Running the Application
Run the script with the following command:

```bash
python app.py
```

### Controls
- **ESC**: Exit the application.
- **C**: Start collecting data for the current label.
- **Z**: Stop collecting and move to the next label.
- **N/K/H**: Switch between logging modes.

## File Structure
```
.
├── app.py
├── model
│   ├── keypoint_classifier
│   │   ├── keypoint_classifier.tflite
│   │   ├── keypoint_classifier_label.csv
│   ├── point_history_classifier
│       ├── point_history_classifier.tflite
│       ├── point_history_classifier_label.csv
├── utils.py
```

## Functionality

1. **Hand Detection**:
   - Uses Mediapipe's hand tracking to detect and extract landmarks.

2. **Data Processing**:
   - Preprocesses landmark and point history data for classification.

3. **Gesture Classification**:
   - Employs custom-trained classifiers to recognize hand gestures.

4. **Response Generation**:
   - Translates recognized gestures into text or audio responses in the user's native language.

5. **Visualization**:
   - Displays hand landmarks, bounding boxes, and gesture information on the video feed.

## Future Upgrades
- **Multi-Language Support**: Expand the language options for text and audio responses.
- **Advanced Gesture Recognition**: Incorporate dynamic gestures and multi-hand interactions.
- **Cloud Integration**: Allow data to be stored and processed on the cloud for improved scalability.
- **Mobile Support**: Develop mobile applications for broader accessibility.
- **Enhanced AI Models**: Integrate state-of-the-art models for higher accuracy and speed.
- **Custom User Profiles**: Enable personalized responses and gesture sets based on user preferences.

## Contributions
Feel free to contribute to this project by adding new gesture types, improving classification accuracy, or enhancing visualization features.

## License
This project is licensed under the MIT License.
