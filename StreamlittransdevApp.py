import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Configuration simple
st.set_page_config(page_title="Détection Casques EPI", page_icon="🪖")
st.title("🪖 Détection de Casques de Sécurité")

# Chargement du modèle (caché)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # Modèle léger

model = load_model()

# Interface simple
option = st.radio("Choisissez une option:", ["📸 Prendre une photo", "📁 Uploader une image"])

if option == "📸 Prendre une photo":
    img_file = st.camera_input("Prenez une photo")
else:
    img_file = st.file_uploader("Choisissez une image", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # Lire l'image
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Détection YOLO
    results = model(cv2_img)
    
    # Annoter l'image
    annotated_img = results[0].plot()
    
    # Afficher résultat
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2_img, channels="BGR", caption="Originale")
    with col2:
        st.image(annotated_img, channels="BGR", caption="Avec détections")
    
    # Compter les personnes
    persons = sum(1 for r in results[0].boxes.cls if model.names[int(r)] == 'person')
    st.success(f"👥 Personnes détectées: {persons}")
    
    # Astuce: Pour détecter les casques, il faudrait un modèle spécialisé
    st.info("💡 Pour détecter spécifiquement les casques, entraînez YOLO sur un dataset de casques de sécurité")
