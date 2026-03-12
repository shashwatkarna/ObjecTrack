from flask import Flask, request, jsonify, render_template
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import torchvision.transforms as transforms
import os
import sys

# Shared Model Path
MODEL_PATH = os.path.join(os.getcwd(), 'best.pt')

def run_flask():
    app = Flask(__name__)
    model = YOLO(MODEL_PATH)

    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0)

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        try:
            img = Image.open(BytesIO(file.read()))
            processed_img = preprocess_image(img)
            results = model(processed_img)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                        confidence = box.conf.item()
                        class_id = int(box.cls.item())
                        name = result.names.get(class_id, "Unknown")
                        detections.append({
                            'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                            'confidence': confidence, 'class': class_id, 'name': name
                        })
            return jsonify(detections)
        except Exception as e:
            return jsonify({'error': f'Prediction error: {str(e)}'})

    @app.route('/')
    def home():
        return render_template('index.html')

    app.run(debug=True, use_reloader=False)

def run_streamlit():
    import streamlit as st
    import cv2
    import numpy as np

    st.set_page_config(page_title="ObjecTrack - Wildlife Detection", page_icon="🐾", layout="wide")

    # Custom CSS
    st.markdown("""
        <style>
        .stApp { background-color: #0f172a; color: #e2e8f0; }
        .brand-header { font-size: 3rem; font-weight: 800; text-align: center; margin-bottom: 2rem; background: -webkit-linear-gradient(#22c55e, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .glass-panel { background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 1rem; padding: 2rem; margin-bottom: 2rem; }
        h2, h3 { color: #ffffff !important; }
        .detection-badge { background-color: rgba(34, 197, 94, 0.2); color: #22c55e; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; border: 1px solid rgba(34, 197, 94, 0.3); }
        .stButton>button { background-color: #16a34a !important; color: white !important; border-radius: 0.75rem !important; border: none !important; padding: 0.5rem 2rem !important; font-weight: 600 !important; transition: all 0.3s ease !important; }
        .stButton>button:hover { background-color: #22c55e !important; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3); }
        </style>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def load_model():
        return YOLO(MODEL_PATH)

    model = load_model()

    st.markdown('<div class="brand-header">Objec<span style="color: #ffffff;">Track</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 3rem;">Detect Wildlife with Precision AR. Powered by YOLOv8.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.subheader("📁 Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

    with col2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.subheader("🔍 Analysis Results")
        if uploaded_file is not None:
            if st.button("Detect Animals"):
                with st.spinner("Running model..."):
                    results = model.predict(image_rgb, imgsz=640)
                    annotated_image = results[0].plot()
                    st.image(annotated_image, caption='Processed Image', use_column_width=True)
                    detections = results[0].boxes
                    if len(detections) > 0:
                        st.markdown(f'<span class="detection-badge">{len(detections)} Detected</span>', unsafe_allow_html=True)
                        for box in detections:
                            class_id = int(box.cls[0])
                            label = results[0].names[class_id]
                            confidence = float(box.conf[0])
                            st.markdown(f"""
                            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                                <div style="font-weight: 600; color: #ffffff;">🐾 {label.capitalize()}</div>
                                <div style="font-size: 0.85rem; color: #10b981;">{confidence:.2%} Match</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No animals detected.")
        else:
            st.markdown('<div style="text-align: center; color: #64748b; padding-top: 5rem;">Upload an image and click Detect.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 5rem; padding: 2rem; border-top: 1px solid rgba(255, 255, 255, 0.05); text-align: center; color: #475569; font-size: 0.85rem;">Made for fun and learning | YOLOv8 | Flask | Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    # Check if running within Streamlit
    try:
        import streamlit as st
        if st.runtime.exists():
            run_streamlit()
        else:
            run_flask()
    except Exception:
        run_flask()
