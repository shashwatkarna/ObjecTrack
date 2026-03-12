from flask import Flask, request, jsonify, render_template
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import torchvision.transforms as transforms

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO(r'F:\2ObjecTrack\working\best.pt') #apna model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # 640x640 or similar sizes
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Load image and preprocess
        img = Image.open(BytesIO(file.read()))
        img = preprocess_image(img)
        
        # Perform inference
        results = model(img)
        
        detections = []
        
        # Process results
        for result in results:
            # Extract bounding boxes
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    try:
                        # Extract box coordinates and other details
                        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                        confidence = box.conf.item()  # Confidence score
                        class_id = int(box.cls.item())  # Class ID
                        name = result.names.get(class_id, "Unknown")  # Class name
                        
                        detections.append({
                            'xmin': xmin,
                            'ymin': ymin,
                            'xmax': xmax,
                            'ymax': ymax,
                            'confidence': confidence,
                            'class': class_id,
                            'name': name
                        })
                    except Exception as e:
                        print(f"Error processing box: {e}")
        
        return jsonify(detections)
    
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': 'An error occurred during prediction'})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
