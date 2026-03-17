import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Détection Casques de Sécurité")
st.write("Upload une image pour détecter les personnes et casques")

uploaded_file = st.file_uploader("Choisir une image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Charger modèle
    model = YOLO('yolov8n.pt')
    
    # Traitement
    image = Image.open(uploaded_file)
    results = model(image)
    
    # Affichage
    st.image(results[0].plot(), caption="Résultat")
    
    # Statistiques
    persons = 0
    for cls in results[0].boxes.cls:
        if model.names[int(cls)] == 'person':
            persons += 1
    
    st.metric("Personnes détectées", persons)
