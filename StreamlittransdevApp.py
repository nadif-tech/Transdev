import streamlit as st
import numpy as np
from PIL import Image
import os

# Configuration de la page
st.set_page_config(
    page_title="Détection EPI",
    page_icon="🪖",
    layout="centered"
)

st.title("🪖 Détection de Casques de Sécurité")
st.write("---")

# Fonction pour charger le modèle YOLO (avec gestion d'erreur)
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Erreur de chargement du modèle: {e}")
        return None

# Interface simple
option = st.radio(
    "Choisissez votre méthode d'entrée:",
    ["📁 Uploader une image", "📸 Prendre une photo"]
)

img_file = None

if option == "📸 Prendre une photo":
    img_file = st.camera_input("Prenez une photo")
else:
    img_file = st.file_uploader(
        "Sélectionnez une image", 
        type=['jpg', 'jpeg', 'png']
    )

if img_file is not None:
    try:
        # Charger le modèle
        with st.spinner("Chargement du modèle YOLO..."):
            model = load_model()
        
        if model is not None:
            # Lire l'image
            image = Image.open(img_file)
            
            # Convertir en RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Faire la prédiction
            with st.spinner("Analyse en cours..."):
                results = model(image)
            
            # Afficher les résultats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Image originale")
                st.image(image, use_columnwidth=True)
            
            with col2:
                st.subheader("🔍 Image avec détections")
                # Annoter l'image
                annotated_image = results[0].plot()
                st.image(annotated_image, use_columnwidth=True)
            
            # Statistiques
            st.subheader("📊 Statistiques")
            
            # Compter les personnes
            persons_count = 0
            detections_list = []
            
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                detections_list.append({
                    'classe': class_name,
                    'confiance': f"{confidence:.2f}"
                })
                
                if class_name == 'person':
                    persons_count += 1
            
            # Afficher les métriques
            col1, col2, col3 = st.columns(3)
            col1.metric("Personnes détectées", persons_count)
            col2.metric("Objets totaux", len(detections_list))
            col3.metric("Confiance moyenne", f"{np.mean([float(d['confiance']) for d in detections_list]):.2f}" if detections_list else "N/A")
            
            # Tableau des détections
            if detections_list:
                st.subheader("📋 Détails des détections")
                st.dataframe(detections_list)
            
            # Message d'information
            st.info("💡 Note: Pour détecter spécifiquement les casques, il faut un modèle entraîné sur des casques de sécurité.")
    
    except Exception as e:
        st.error(f"Une erreur est survenue: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Commencez par uploader une image ou prendre une photo")

# Pied de page
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Application de détection d'objets avec YOLOv8</p>
    </div>
""", unsafe_allow_html=True)
