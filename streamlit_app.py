import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('yolov8s.pt')

model = load_model()

# Load class names
@st.cache_data
def load_class_names():
    with open("coco.txt", "r") as f:
        class_list = f.read().split("\n")
    return class_list

class_list = load_class_names()

# Define parking areas
area1 = [(52,364),(30,417),(73,412),(88,369)]
area2 = [(105,353),(86,428),(137,427),(146,358)]
area3 = [(159,354),(150,427),(204,425),(203,353)]
area4 = [(217,352),(219,422),(273,418),(261,347)]
area5 = [(274,345),(286,417),(338,415),(321,345)]
area6 = [(336,343),(357,410),(409,408),(382,340)]
area7 = [(396,338),(426,404),(479,399),(439,334)]
area8 = [(458,333),(494,397),(543,390),(495,330)]
area9 = [(511,327),(557,388),(603,383),(549,324)]
area10 = [(564,323),(615,381),(654,372),(596,315)]
area11 = [(616,316),(666,369),(703,363),(642,312)]
area12 = [(674,311),(730,360),(764,355),(707,308)]

areas = [area1, area2, area3, area4, area5, area6, 
         area7, area8, area9, area10, area11, area12]

def process_frame(frame):
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    # Initialize counters for each area
    counts = [0] * 12
    
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), row[4], int(row[5])
        c = class_list[d]
        
        if 'car' in c:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            for i, area in enumerate(areas):
                if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                    counts[i] += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    break
    
    # Draw parking areas
    for i, (area, count) in enumerate(zip(areas, counts)):
        color = (0, 0, 255) if count >= 1 else (0, 255, 0)
        cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
        position = area[0]  # Use first point as text position
        cv2.putText(frame, str(i+1), position, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    
    occupied = sum(counts)
    free = 12 - occupied
    
    return frame, occupied, free

def main():
    st.title("Parking Space Detection System")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save uploaded file to temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Open video file
        cap = cv2.VideoCapture(tfile.name)
        
        # Button to check parking spaces
        if st.button("Check Parking Spaces"):
            # Get one frame for analysis
            ret, frame = cap.read()
            if ret:
                processed_frame, occupied, free = process_frame(frame)
                
                # Display results
                st.image(processed_frame, channels="BGR", caption="Processed Frame")
                st.write(f"Occupied spaces: {occupied}")
                st.write(f"Free spaces: {free}")
                
                # Show detailed status
                st.subheader("Parking Space Status")
                cols = st.columns(4)
                for i in range(12):
                    with cols[i % 4]:
                        st.write(f"Space {i+1}: {'Occupied' if occupied > i else 'Free'}")
            else:
                st.error("Could not read video frame")
        
        # Clean up
        cap.release()
        os.unlink(tfile.name)

if __name__ == "__main__":
    main()
