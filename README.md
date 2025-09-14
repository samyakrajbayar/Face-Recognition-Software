# Advanced Face Recognition Software

A comprehensive Python-based face recognition system with detection, identification, and verification capabilities. Supports real-time webcam feeds, image processing, video analysis, and includes both CLI and GUI interfaces.

## Features

### Core Functionality
- **Face Detection**: Detect multiple faces in images, videos, and live webcam feeds
- **Face Recognition**: Identify known faces from a pre-stored database
- **Face Verification**: 1-to-1 matching for authentication purposes
- **User Registration**: Easy enrollment of new faces with automatic encoding
- **Multi-interface Support**: Both command-line and graphical user interfaces

### Advanced Features
- **Real-time Processing**: Live webcam recognition with bounding boxes and labels
- **Attendance System**: Automatic attendance logging with CSV export
- **Database Storage**: SQLite database for efficient face encoding storage
- **Configurable Tolerance**: Adjustable recognition sensitivity
- **Batch Processing**: Process multiple images and videos
- **Performance Optimized**: Frame resizing and efficient matching algorithms

### Technical Capabilities
- Multiple face detection in single frame
- Confidence scoring for recognition results
- Error handling for edge cases
- Modular architecture for easy extension
- Cross-platform compatibility

## Installation

### Prerequisites
- Python 3.7 or higher
- Webcam (for real-time recognition)
- CMake (for dlib compilation)

### Step 1: Clone Repository
```bash
git clone https://github.com/your-repo/face-recognition-software.git
cd face-recognition-software
```

### Step 2: Install Dependencies

#### Option A: Using pip
```bash
pip install -r requirements.txt
```

#### Option B: Manual Installation
```bash
pip install face-recognition opencv-python numpy Pillow
```

#### Option C: For Development
```bash
pip install -r requirements.txt
pip install pytest black flake8  # Development tools
```

### Step 3: System-specific Setup

#### Windows
```bash
# Install Visual Studio Build Tools if needed
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019
```

#### macOS
```bash
brew install cmake
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev
sudo apt-get install libx11-dev libgtk-3-dev
```

## Usage

### GUI Interface (Recommended)

Launch the graphical interface:
```bash
python face_recognition_system.py --gui
```

#### GUI Features:
- **Start Webcam Recognition**: Begin real-time face recognition
- **Register New Face**: Add new faces to the database
- **Process Image**: Analyze single images
- **Process Video**: Analyze video files
- **Settings Panel**: Adjust recognition tolerance and enable attendance logging
- **Status Monitor**: Real-time logging and system status

### Command Line Interface

#### Basic Commands

**Start webcam recognition:**
```bash
python face_recognition_system.py --webcam
```

**Process single image:**
```bash
python face_recognition_system.py --image path/to/image.jpg
```

**Process video file:**
```bash
python face_recognition_system.py --video path/to/video.mp4
```

**Register new face:**
```bash
python face_recognition_system.py --register "John Doe" path/to/john.jpg
```

#### Advanced Options

**Set custom tolerance:**
```bash
python face_recognition_system.py --webcam --tolerance 0.5
```

**Enable attendance logging:**
```bash
python face_recognition_system.py --webcam --attendance
```

**Combine multiple options:**
```bash
python face_recognition_system.py --webcam --tolerance 0.4 --attendance
```

## Configuration

### Recognition Tolerance
- **Range**: 0.0 - 1.0
- **Default**: 0.6
- **Lower values**: More strict matching (fewer false positives)
- **Higher values**: More lenient matching (fewer false negatives)

### File Structure
```
face-recognition-software/
├── face_recognition_system.py    # Main application
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
├── faces.db                      # SQLite database (auto-created)
├── attendance.csv                # Attendance logs (auto-created)
├── test_images/                  # Sample images for testing
│   ├── person1.jpg
│   ├── person2.jpg
│   └── group_photo.jpg
└── examples/                     # Usage examples
    ├── batch_processing.py
    └── custom_integration.py
```

## Database Schema

The SQLite database stores face encodings with the following structure:

```sql
CREATE TABLE faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    encoding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## API Reference

### FaceRecognitionEngine Class

#### Methods

**`__init__(tolerance=0.6)`**
- Initialize the face recognition engine
- `tolerance`: Recognition sensitivity (0.0-1.0)

**`detect_faces(image)`**
- Detect faces in an image
- Returns: face locations and encodings

**`identify_face(face_encoding)`**
- Identify a face from known faces database
- Returns: (name, confidence) tuple

**`verify_face(face_encoding, name)`**
- Verify if face matches specific person
- Returns: (is_match, confidence) tuple

**`register_face(image, name)`**
- Register new face in database
- Returns: success boolean

**`process_frame(frame, log_attendance=False)`**
- Process single frame/image
- Returns: processed frame with annotations

### FaceDatabase Class

**`save_face(name, encoding)`**
- Save face encoding to database

**`load_faces()`**
- Load all faces from database

**`delete_face(name)`**
- Remove face from database

### AttendanceLogger Class

**`log_attendance(name, status='Present')`**
- Log attendance for recognized person

## Examples

### Basic Face Recognition Script

```python
import cv2
from face_recognition_system import FaceRecognitionEngine

# Initialize engine
engine = FaceRecognitionEngine(tolerance=0.6)

# Register a face
image = cv2.imread('person.jpg')
engine.register_face(image, 'John Doe')

# Process webcam feed
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        processed = engine.process_frame(frame)
        cv2.imshow('Recognition', processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

### Custom Integration Example

```python
from face_recognition_system import FaceRecognitionEngine

class SecuritySystem:
    def __init__(self):
        self.engine = FaceRecognitionEngine(tolerance=0.5)
        self.authorized_users = ['John Doe', 'Jane Smith']
    
    def authenticate_user(self, image):
        locations, encodings = self.engine.detect_faces(image)
        
        for encoding in encodings:
            name, confidence = self.engine.identify_face(encoding)
            if name in self.authorized_users and confidence > 70:
                return True, name
        
        return False, None

# Usage
security = SecuritySystem()
image = cv2.imread('access_attempt.jpg')
authorized, user = security.authenticate_user(image)

if authorized:
    print(f"Access granted for {user}")
else:
    print("Access denied")
```

## Performance Optimization

### Speed Improvements

1. **Frame Resizing**: Process smaller frames for real-time recognition
```python
# Resize frame to 1/4 size for faster processing
small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
```

2. **Process Every N Frames**: Skip frames for better performance
```python
frame_count = 0
if frame_count % 3 == 0:  # Process every 3rd frame
    # Perform recognition
```

3. **GPU Acceleration** (Optional):
```bash
# Install GPU-accelerated OpenCV
pip uninstall opencv-python
pip install opencv-contrib-python

# Enable CUDA for dlib (requires CUDA toolkit)
```

### Memory Optimization

1. **Batch Processing**: Process multiple faces at once
2. **Encoding Caching**: Store computed encodings
3. **Database Indexing**: Optimize database queries

## Troubleshooting

### Common Issues

**1. "No module named 'face_recognition'"**
```bash
# Solution: Install with specific version
pip install face-recognition==1.3.0
```

**2. "CMake not found"**
```bash
# Windows: Install Visual Studio Build Tools
# macOS: brew install cmake
# Linux: sudo apt-get install cmake
```

**3. "Could not open camera"**
```python
# Check camera index
cap = cv2.VideoCapture(1)  # Try different indices
```

**4. "No faces detected"**
- Ensure good lighting
- Face should be clearly visible
- Try adjusting tolerance
- Check image quality

**5. Performance Issues**
- Reduce frame size
- Lower video resolution
- Process fewer frames per second
- Close other applications

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Use Cases

### 1. Office Attendance System
```python
# Configure for attendance tracking
engine = FaceRecognitionEngine()
# Enable attendance logging in GUI or use --attendance flag
```

### 2. Security Access Control
```python
# High security settings
engine = FaceRecognitionEngine(tolerance=0.4)  # Strict matching
```

### 3. Photo Organization
```python
# Process photo albums
for image_file in photo_directory:
    image = cv2.imread(image_file)
    processed = engine.process_frame(image)
    # Save with recognized names
```

### 4. Event Check-in System
```python
# Real-time event registration
# Use GUI with attendance logging enabled
```

## Advanced Features

### Custom Models

For enhanced accuracy, you can integrate custom models:

```python
# Example: Mask detection integration
import tensorflow as tf

class EnhancedFaceEngine(FaceRecognitionEngine):
    def __init__(self):
        super().__init__()
        self.mask_model = tf.keras.models.load_model('mask_detection.h5')
    
    def detect_mask(self, face_image):
        # Custom mask detection logic
        prediction = self.mask_model.predict(face_image)
        return prediction > 0.5
```

### Cloud Integration

```python
# Example: AWS Rekognition comparison
import boto3

def compare_with_aws(image_bytes):
    rekognition = boto3.client('rekognition')
    response = rekognition.detect_faces(
        Image={'Bytes': image_bytes},
        Attributes=['ALL']
    )
    return response
```

## Testing

### Unit Tests
```bash
# Run tests
python -m pytest tests/

# Test specific functionality
python -m pytest tests/test_face_detection.py
```

### Sample Test Images

Include these test scenarios:
- Single face images
- Multiple face images  
- Poor lighting conditions
- Different angles
- Masked faces
- Group photos

## Deployment

### Standalone Executable

Create executable using PyInstaller:
```bash
pip install pyinstaller
pyinstaller --onefile --windowed face_recognition_system.py
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "face_recognition_system.py", "--gui"]
```

### Web API Version

```python
from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np

app = Flask(__name__)
engine = FaceRecognitionEngine()

@app.route('/recognize', methods=['POST'])
def recognize_face():
    # Decode base64 image
    image_data = base64.b64decode(request.json['image'])
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process image
    locations, encodings = engine.detect_faces(image)
    results = []
    
    for encoding in encodings:
        name, confidence = engine.identify_face(encoding)
        results.append({
            'name': name,
            'confidence': confidence
        })
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Security Considerations

1. **Data Privacy**: Face encodings are stored locally
2. **Access Control**: Implement user authentication for sensitive applications
3. **Encryption**: Consider encrypting the database file
4. **Network Security**: Use HTTPS for web deployments
5. **Audit Logging**: Track access and modifications

## Performance Benchmarks

### Hardware Requirements

**Minimum:**
- CPU: Intel i3 or AMD equivalent
- RAM: 4GB
- Storage: 1GB free space

**Recommended:**
- CPU: Intel i5 or AMD equivalent
- RAM: 8GB
- GPU: NVIDIA GTX 1050 or better (for GPU acceleration)
- Storage: 2GB free space

### Expected Performance

- **Real-time Recognition**: 15-30 FPS (depending on hardware)
- **Batch Processing**: 50-100 images per minute
- **Database Capacity**: 10,000+ faces
- **Recognition Accuracy**: 95-99% under good conditions

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-repo/face-recognition-software.git
cd face-recognition-software

# Create virtual environment
python -m venv face_recognition_env
source face_recognition_env/bin/activate  # Linux/Mac
# face_recognition_env\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest

# Format code
black face_recognition_system.py

# Lint code
flake8 face_recognition_system.py
```

## License

MIT License - see LICENSE file for details

## Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the examples directory

## Changelog

### Version 1.0.0
- Initial release with core functionality
- GUI and CLI interfaces
- SQLite database integration
- Attendance logging system



- [face_recognition library](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [dlib](http://dlib.net/) for machine learning algorithms
