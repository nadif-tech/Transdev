import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Configuration
st.set_page_config(page_title="Détection Casques EPI", page_icon="🪖")
st.title("🪖 Détection de Casques de Sécurité")

# Chargement du modèle
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Interface simple
option = st.radio("Choisissez une option:", ["📸 Prendre une photo", "📁 Uploader une image"])

if option == "📸 Prendre une photo":
    img_file = st.camera_input("Prenez une photo")
else:
    img_file = st.file_uploader("Choisissez une image", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    # Lire l'image avec PIL (pas besoin de cv2)
    image = Image.open(img_file)
    
    # Convertir en numpy array pour YOLO
    img_array = np.array(image)
    
    # Détection YOLO
    results = model(img_array)
    
    # Annoter l'image
    annotated_img = results[0].plot()
    
    # Afficher résultats
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Image originale")
    with col2:
        st.image(annotated_img, caption="Avec détections")
    
    # Compter les personnes
    persons = 0
    if hasattr(results[0].boxes, 'cls'):
        persons = sum(1 for cls in results[0].boxes.cls if model.names[int(cls)] == 'person')
    
    st.success(f"👥 Personnes détectées: {persons}")
    
    # Afficher tous les objets détectés
    st.subheader("📋 Détails des détections:")
    for i, (box, cls, conf) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf)):
        class_name = model.names[int(cls)]
        st.write(f"{i+1}. {class_name} - Confiance: {conf:.2f}")
