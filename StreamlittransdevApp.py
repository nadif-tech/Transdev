import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Configuration de la page
st.set_page_config(
    page_title="Détection Casques EPI",
    page_icon="🪖",
    layout="wide"
)

st.title("🪖 Détection de Casques de Sécurité")
st.markdown("---")

# Vérification de YOLO
try:
    from ultralytics import YOLO
    model_available = True
    st.sidebar.success("✅ YOLO chargé")
    
    @st.cache_resource
    def load_model():
        return YOLO('yolov8n.pt')
    model = load_model()
except:
    model_available = False
    st.sidebar.warning("⚠️ Mode: Détection de visages uniquement")
    model = None

# Interface
st.subheader("📸 Prendre une photo")
img_file = st.camera_input("Cliquez pour prendre une photo")

if img_file is not None:
    # Lire l'image
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    with st.spinner("Analyse..."):
        if model_available:
            # YOLO
            results = model(cv2_img)
            annotated_img = results[0].plot()
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(rgb_img, caption="Originale", use_column_width=True)
            with col2:
                st.image(annotated_img, channels="BGR", caption="Détections YOLO", use_column_width=True)
            
            # Stats
            persons = 0
            for box in results[0].boxes:
                if model.names[int(box.cls[0])] == 'person':
                    persons += 1
            st.metric("Personnes détectées", persons)
            
        else:
            # OpenCV visages
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(cv2_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            st.image(cv2_img, channels="BGR", caption=f"{len(faces)} visage(s) détecté(s)", use_column_width=True)
            st.metric("Personnes détectées", len(faces))

# Instructions simples
with st.expander("ℹ️ Aide"):
    st.write("1. Cliquez sur 'Prenez une photo'")
    st.write("2. Autorisez l'accès à la caméra")
    st.write("3. La photo sera analysée automatiquement")
    
    if model_available:
        st.write("✅ Mode YOLO actif")
    else:
        st.write("⚠️ Mode visages OpenCV actif")
        st.write("Pour YOLO, ajoutez 'ultralytics' dans requirements.txt")

# Pied de page
st.markdown("---")
mode = "YOLO" if model_available else "OpenCV Visages"
st.markdown(f"<p style='text-align: center'>Mode: {mode}</p>", unsafe_allow_html=True)
