import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import threading

# Configuration de la page
st.set_page_config(
    page_title="Détection EPI en temps réel",
    page_icon="🎥",
    layout="wide"
)

st.title("🪖 Détection de Casques de Sécurité - Caméra en direct")
st.markdown("---")

# Chargement du modèle
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    st.info("📹 Utilisez votre caméra pour détecter les personnes en temps réel")

# Option 1: Utilisation de streamlit-webrtc (RECOMMANDÉ pour Streamlit Cloud)
st.subheader("📹 Détection en temps réel avec WebRTC")

try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    
    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.model = model
            self.conf_threshold = confidence_threshold
        
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Détection YOLO
            results = self.model(img, conf=self.conf_threshold)
            
            # Annoter l'image
            annotated_img = results[0].plot()
            
            return annotated_img
    
    webrtc_streamer(
        key="object-detection",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
except ImportError:
    st.warning("streamlit-webrtc n'est pas installé. Installation en cours...")
    st.code("pip install streamlit-webrtc av")

# Option 2: Alternative avec OpenCV (pour test local)
st.subheader("🔄 Alternative avec OpenCV (pour test local)")

use_opencv = st.checkbox("Utiliser OpenCV (test local uniquement)")

if use_opencv:
    st.warning("⚠️ Cette option fonctionne mieux en local, pas sur Streamlit Cloud")
    
    # Bouton pour démarrer/arrêter
    run = st.checkbox("Démarrer la caméra")
    
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Impossible d'ouvrir la caméra")
        else:
            st.success("Caméra démarrée! Appuyez sur le checkbox pour arrêter")
            
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Erreur de capture")
                    break
                
                # Détection
                results = model(frame, conf=confidence_threshold)
                annotated_frame = results[0].plot()
                
                # Convertir BGR en RGB
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Afficher
                FRAME_WINDOW.image(annotated_frame_rgb)
            
            cap.release()
    else:
        st.info("Cliquez sur 'Démarrer la caméra' pour commencer")

# Option 3: Capture photo avec caméra
st.subheader("📸 Capture photo avec caméra")

img_file = st.camera_input("Prenez une photo")

if img_file is not None:
    # Lire l'image
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Détection
    with st.spinner("Analyse de la photo..."):
        results = model(cv2_img, conf=confidence_threshold)
        annotated_img = results[0].plot()
    
    # Afficher
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2_img, channels="BGR", caption="Photo originale")
    with col2:
        st.image(annotated_img, channels="BGR", caption="Avec détections")
    
    # Statistiques
    persons = sum(1 for r in results[0].boxes.cls if model.names[int(r)] == 'person')
    st.success(f"👥 Personnes détectées: {persons}")
    
    # Détails
    st.subheader("📋 Détections:")
    for i, (box, cls, conf) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf)):
        class_name = model.names[int(cls)]
        st.write(f"{i+1}. {class_name} - Confiance: {conf:.2f}")

# Instructions
with st.expander("📖 Instructions"):
    st.markdown("""
    ### Comment utiliser l'application
    
    1. **Pour la caméra en temps réel**: 
       - Utilisez la section WebRTC (recommandé pour Streamlit Cloud)
       - Autorisez l'accès à votre caméra quand le navigateur le demande
    
    2. **Pour capturer une photo**:
       - Utilisez la section "Capture photo avec caméra"
       - Cliquez sur "Prenez une photo"
    
    3. **Paramètres**:
       - Ajustez le seuil de confiance dans la barre latérale
    
    ### Notes importantes
    - Le modèle détecte les personnes (pour les casques, il faudrait un modèle spécifique)
    - La détection fonctionne mieux avec un bon éclairage
    - Pour Streamlit Cloud, la méthode WebRTC est plus fiable
    """)

# Pied de page
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Application de détection en temps réel avec YOLOv8</p>
    </div>
""", unsafe_allow_html=True)
