import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="🪖 Détection Casque EPI", layout="wide")
st.title("🪖 Détection Casque EPI - YOLOv8 Nano + Webcam")

model = YOLO("yolov8n.pt")

run = st.checkbox("Démarrer la caméra")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Erreur caméra")
        break
    frame = cv2.resize(frame, (640, 480))
    results = model(frame)
    annotated_frame = results[0].plot()
    FRAME_WINDOW.image(annotated_frame, channels="BGR")

camera.release()
