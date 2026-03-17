import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

st.title("🎥 Détection en temps réel")

# Charger modèle
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Capture avec caméra
img_file = st.camera_input("Prenez une photo")

if img_file is not None:
    # Lire l'image
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Détection
    results = model(cv2_img)
    annotated_img = results[0].plot()
    
    # Afficher
    st.image(annotated_img, channels="BGR", caption="Résultat")
    
    # Compter personnes
    persons = sum(1 for r in results[0].boxes.cls if model.names[int(r)] == 'person')
    st.metric("Personnes détectées", persons)
