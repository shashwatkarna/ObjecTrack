# ObjecTrack: Animal Object Detection System using YOLOv8

A Flask-based web application for real-time animal detection and classification using YOLOv8 deep learning model.

## Project Overview

This project implements a **YOLO v8 object detection system** trained to detect and classify **12 different animal species**:
- Bird
- Butterfly
- Cat
- Cheetah
- Cow
- Dog
- Elephant
- Fish
- Fox
- Horse
- Sheep
- Squirrel

## Project Development Pipeline

### 1. Dataset Collection
The project uses a custom animal dataset structured as follows:
- **Raw Images**: Collected from multiple sources (online datasets, webcam, manual collection)
- **Total Images**: Distributed across train/val/test splits
- **Location**: `data/train/images`, `data/val/images`, `data/test/images`

### 2. Data Annotation & Labelling
- **Tool Used**: Roboflow or LabelImg for bounding box annotations
- **Format**: YOLO format (`.txt` files with class ID and normalized coordinates)
- **Labels Location**: `data/train/labels`, `data/val/labels`, `data/test/labels`
- **Each Label File Contains**: One detection per line with format: `class_id center_x center_y width height` (normalized values 0-1)

### 3. Dataset Configuration
- **File**: `data.yaml`
- **Structure**:
  ```yaml
  train: path/to/train/images
  val: path/to/val/images
  test: path/to/test/images
  nc: 12  # Number of classes
  names: ['bird', 'butterfly', 'cat', ...]  # Class names
  ```

### 4. Model Training
- **Framework**: YOLOv8 (Ultralytics)
- **Base Model**: YOLOv8n (nano) for lightweight inference
- **Training Approach**: Transfer learning on custom dataset
- **Training Command**:
  ```bash
  from ultralytics import YOLO
  model = YOLO('yolov8n.pt')
  results = model.train(data='data.yaml', epochs=100, imgsz=640)
  ```
- **Output**: Trained model saved as `best.pt`

### 5. Model Files
- `best.pt` - Final trained model (best validation performance)
- `yolov8n.pt` - Pre-trained YOLOv8 nano baseline
- Located in: `working/`, `Ani_det/`, `streamlit_app/` folders

## Project Structure

```
ObjecTrack/
│
├── 📄 app.py                      # Main Flask web application
├── 📄 requirements.txt            # Python dependencies
├── 📄 README.md                   # Project documentation
├── 📄 data.yaml                   # Dataset configuration
├── 📄 .gitignore                  # Git ignore rules
│
├── 📁 models/                     # Trained model weights
│   └── best.pt                    # YOLOv8 trained model
│
├── 📁 data/                       # Dataset folder (70-10-20 split)
│   ├── 📁 train/                  # Training data (70%)
│   │   ├── images/               # Training images
│   │   └── labels/               # Training annotations (YOLO format)
│   ├── 📁 val/                    # Validation data (10%)
│   │   ├── images/               # Validation images
│   │   └── labels/               # Validation annotations
│   └── 📁 test/                   # Test data (20%)
│       ├── images/               # Test images
│       └── labels/               # Test annotations
│
├── 📁 templates/                  # HTML templates
│   └── index.html                # Main web interface
│
├── 📁 static/                     # Static web assets
│   ├── styles.css                # CSS styling
│   └── scripts.js                # JavaScript functionality
│
└── 📁 runs/                       # Training outputs (auto-generated)
    └── weights/
        └── best.pt               # Best model checkpoint
```

## Installation

### Prerequisites
- Python 3.8+ (recommended: Python 3.10 or 3.11)
- pip package manager
- Virtual environment (recommended)

### Setup Steps

1. **Clone/Navigate to project directory**:
   ```bash
   cd f:\2ObjecTrack\working
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Upgrade pip**:
   ```bash
   python -m pip install --upgrade pip
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies
- **flask** (>=3.0.0) - Web framework
- **Pillow** (>=9.0.0) - Image processing
- **ultralytics** (>=8.0.0) - YOLOv8 library
- **torch** (>=2.0.0) - PyTorch deep learning framework
- **torchvision** (>=0.15.0) - Computer vision utilities
- **numpy** (>=1.20.0) - Numerical computing

## Usage

### Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   - Open your browser and navigate to: `http://localhost:5000`

3. **Make predictions**:
   - Upload an animal image using the web form
   - Click "Predict" button
   - Results display bounding boxes with:
     - Class name (detected animal)
     - Confidence score (0-1)
     - Bounding box coordinates (xmin, ymin, xmax, ymax)

### API Endpoint

**POST** `/predict`
- **Parameter**: `file` (image file upload)
- **Response**: JSON with detections array
  ```json
  [
    {
      "xmin": 50,
      "ymin": 100,
      "xmax": 200,
      "ymax": 250,
      "confidence": 0.92,
      "class": 2,
      "name": "cat"
    }
  ]
  ```

### Python Script Usage

```python
from ultralytics import YOLO
from PIL import Image

# Load the trained model
model = YOLO('best.pt')

# Predict on image
results = model('path/to/image.jpg')

# Extract detections
for result in results:
    boxes = result.boxes
    for box in boxes:
        print(f"Class: {result.names[int(box.cls)]}, Confidence: {box.conf:.2f}")
```

## Training Your Own Model

### 1. Prepare Dataset
- Organize images in `data/train/images`, `data/val/images`, `data/test/images`
- Annotate with bounding boxes (use Roboflow, LabelImg, or similar tools)
- Export labels in YOLO format to respective `labels/` folders

### 2. Create data.yaml
```yaml
train: path/to/train/images
val: path/to/val/images
test: path/to/test/images
nc: <number_of_classes>
names: [<class1>, <class2>, ...]
```

### 3. Run Training Script
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train on custom dataset
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    device=0,  # GPU device ID (0 for first GPU)
    patience=20  # Early stopping
)

# Save best model
best_model = YOLO('runs/detect/train/weights/best.pt')
```

## Model Performance

- **Classes**: 12 animal species
- **Architecture**: YOLOv8 Nano (lightweight, fast inference)
- **Input Size**: 640x640 pixels
- **Inference Speed**: ~5-10ms per image (GPU dependent)
- **Dataset**: Custom annotated animal images

## Troubleshooting

### Model File Not Found
- Ensure `best.pt` exists in the working directory
- Update model path in `app.py` if model is in different location:
  ```python
  model = YOLO('path/to/best.pt')
  ```

### Port Already in Use
- Change Flask port in `app.py`:
  ```python
  app.run(debug=True, port=5001)
  ```

### CUDA/GPU Issues
- If GPU not available, model will use CPU (slower inference)
- Install CPU-only PyTorch if GPU version causes issues:
  ```bash
  pip install torch torchvision -f https://download.pytorch.org/whl/cpu/torch_stable.html
  ```

## Future Enhancements

- [ ] Real-time video stream detection
- [ ] Batch image processing
- [ ] Model quantization for edge deployment
- [ ] Mobile app integration
- [ ] Database for prediction history
- [ ] Advanced filtering and confidence threshold UI

## License

This project uses YOLOv8 by Ultralytics (licensed under AGPL-3.0).

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Created**: February 2026  
**Last Updated**: February 22, 2026
