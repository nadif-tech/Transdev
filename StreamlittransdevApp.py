import streamlit as st
import sys
import subprocess

# Vérifier et installer ultralytics si nécessaire
try:
    from ultralytics import YOLO
    st.success("✅ Ultralytics importé avec succès")
except ImportError as e:
    st.warning("⚠️ Ultralytics non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO
    st.success("✅ Ultralytics installé et importé")

import cv2
import numpy as np
from PIL import Image
import tempfile
import time

# Configuration de la page
st.set_page_config(
    page_title="Détection EPI",
    page_icon="🪖",
    layout="wide"
)

st.title("🪖 Détection de Casques de Sécurité")
st.markdown("---")

# Chargement du modèle
@st.cache_resource
def load_model():
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle: {e}")
        return None

model = load_model()

if model is None:
    st.error("Impossible de charger le modèle YOLO")
    st.stop()

# Interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Capture Caméra")
    img_file = st.camera_input("Prenez une photo")

with col2:
    st.subheader("📁 Upload Image")
    uploaded_file = st.file_uploader("Ou uploader une image", type=['jpg', 'jpeg', 'png'])

# Traitement de l'image
img_to_process = None
source = None

if img_file is not None:
    img_to_process = img_file
    source = "caméra"
elif uploaded_file is not None:
    img_to_process = uploaded_file
    source = "upload"

if img_to_process is not None:
    # Lire l'image
    bytes_data = img_to_process.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Détection
    with st.spinner("Analyse en cours..."):
        results = model(cv2_img)
        annotated_img = results[0].plot()
    
    # Affichage
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2_img, channels="BGR", caption=f"Image originale ({source})")
    with col2:
        st.image(annotated_img, channels="BGR", caption="Avec détections")
    
    # Statistiques
    st.subheader("📊 Résultats")
    
    persons = 0
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        
        st.write(f"- **{class_name}** - Confiance: {confidence:.2f}")
        
        if class_name == 'person':
            persons += 1
    
    st.metric("👥 Personnes détectées", persons)

# Instructions
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    1. **Capture caméra**: Cliquez sur "Prenez une photo" pour capturer
    2. **Upload**: Uploader une image depuis votre ordinateur
    3. **Résultats**: Les détections apparaîtront automatiquement
    """)
