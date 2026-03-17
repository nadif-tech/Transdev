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

# Vérification simple de la disponibilité de YOLO
try:
    from ultralytics import YOLO
    model_available = True
    st.sidebar.success("✅ YOLO chargé avec succès")
    
    # Chargement du modèle
    @st.cache_resource
    def load_model():
        return YOLO('yolov8n.pt')
    
    model = load_model()
    
except ImportError:
    model_available = False
    st.sidebar.warning("⚠️ Mode dégradé : Détection par visages uniquement")
    st.sidebar.info("Pour activer YOLO, ajoutez 'ultralytics' dans requirements.txt")
    model = None

# Interface principale
st.subheader("📸 Prendre une photo avec la caméra")
img_file = st.camera_input("Cliquez pour prendre une photo")

# Option upload
st.subheader("📁 Ou uploader une image")
uploaded_file = st.file_uploader("Choisir un fichier", type=['jpg', 'jpeg', 'png'])

# Déterminer quelle image utiliser
image_to_process = None
source = ""

if img_file is not None:
    image_to_process = img_file
    source = "caméra"
elif uploaded_file is not None:
    image_to_process = uploaded_file
    source = "upload"

# Traitement de l'image
if image_to_process is not None:
    # Lire l'image
    bytes_data = image_to_process.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    with st.spinner("Analyse en cours..."):
        
        if model_available and model is not None:
            # Détection YOLO
            results = model(cv2_img)
            annotated_img = results[0].plot()
            
            # Affichage
            col1, col2 = st.columns(2)
            with col1:
                st.image(rgb_img, caption=f"Image originale ({source})", use_column_width=True)
            with col2:
                st.image(annotated_img, channels="BGR", caption="Avec détections YOLO", use_column_width=True)
            
            # Statistiques
            st.subheader("📊 Résultats de la détection")
            
            persons = 0
            detections = []
            
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                detections.append({
                    "Objet": class_name,
                    "Confiance": f"{confidence:.2%}"
                })
                
                if class_name == 'person':
                    persons += 1
            
            # Métriques
            col1, col2, col3 = st.columns(3)
            col1.metric("👥 Personnes", persons)
            col2.metric("📦 Objets", len(detections))
            
            # Tableau des détections
            if detections:
                st.dataframe(detections)
            
        else:
            # Détection alternative avec OpenCV (visages)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Dessiner les rectangles
            img_with_faces = cv2_img.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(img_with_faces, "Personne", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Affichage
            col1, col2 = st.columns(2)
            with col1:
                st.image(rgb_img, caption=f"Image originale ({source})", use_column_width=True)
            with col2:
                st.image(img_with_faces, channels="BGR", caption="Détection de visages", use_column_width=True)
            
            # Statistiques
            st.subheader("📊 Résultats")
            st.metric("👥 Personnes détectées", len(faces))
            
            if len(faces) > 0:
                st.success(f"✅ {len(faces)} personne(s) détectée(s)")
            else:
                st.info("ℹ️ Aucune personne détectée")
            
            # Message d'information
            st.info("💡 Conseil: Pour une meilleure détection, ajoutez 'ultralytics' dans requirements.txt")

else:
    st.info("👆 Prenez une photo ou uploadez une image pour commencer")

# Instructions et informations sur les dépendances
with st.expander("ℹ️ Informations et dépannage"):
    st.markdown("""
    ### 📋 Instructions
    
    1. **Pour utiliser la caméra**:
       - Cliquez sur "Prenez une photo"
       - Autorisez l'accès à la caméra
       - La photo sera automatiquement analysée
    
    2. **Pour uploader une image**:
       - Cliquez sur "Browse files" dans la section upload
       - Sélectionnez une image depuis votre ordinateur
    
    ### 🔧 Mode de détection actuel
    
    - **Si YOLO est installé**: Détection complète de tous les objets
    - **Si YOLO n'est pas installé**: Détection des visages avec OpenCV
    
    ### 📦 Dépendances requises
    
    Pour activer YOLO, votre fichier `requirements.txt` doit contenir:
