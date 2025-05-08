import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import streamlit as st
import tempfile

# Initialize session state
if 'parking_data' not in st.session_state:
    st.session_state.parking_data = {
        'occupied': 0,
        'free': 12,
        'spaces': [False]*12  # False=free, True=occupied
    }

# Load model and classes (cached)
@st.cache_resource
def load_model():
    model = YOLO('yolov8s.pt')
    with open("coco.txt", "r") as f:
        class_list = f.read().split("\n")
    return model, class_list

model, class_list = load_model()

# Define parking areas
parking_areas = [
    [(52,364),(30,417),(73,412),(88,369)],   # Space 1
    [(105,353),(86,428),(137,427),(146,358)], # Space 2
    # ... add all 12 areas here
]

def process_frame(frame):
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    counts = [0] * 12
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            if class_list[cls] == 'car':
                cx, cy = (x1+x2)//2, (y1+y2)//2
                for i, area in enumerate(parking_areas):
                    if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                        counts[i] += 1
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    
    # Update parking data
    occupied = sum(min(1, count) for count in counts)  # Max 1 car per space
    st.session_state.parking_data = {
        'occupied': occupied,
        'free': 12 - occupied,
        'spaces': [count > 0 for count in counts]
    }
    
    # Draw parking spaces
    for i, area in enumerate(parking_areas):
        color = (0,0,255) if st.session_state.parking_data['spaces'][i] else (0,255,0)
        cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
        cv2.putText(frame, str(i+1), area[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    return frame

def main():
    st.title("🚗 Smart Parking Detection")
    
    uploaded_file = st.file_uploader("Upload parking lot video", type=["mp4", "mov"])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        if st.button("Analyze Parking Spaces"):
            cap = cv2.VideoCapture(tfile.name)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                processed_frame = process_frame(frame)
                st.image(processed_frame, channels="BGR", use_column_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Free Spaces", st.session_state.parking_data['free'])
                with col2:
                    st.metric("Occupied Spaces", st.session_state.parking_data['occupied'])
                
                st.subheader("Space Status")
                cols = st.columns(4)
                for i in range(12):
                    with cols[i%4]:
                        st.write(f"Space {i+1}:", 
                                "🟢 Free" if not st.session_state.parking_data['spaces'][i] else "🔴 Occupied")
            else:
                st.error("Couldn't read video frame")

if __name__ == "__main__":
    main()
