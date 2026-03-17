import streamlit as st
from PIL import Image
import numpy as np
import time

# Configuration de la page
st.set_page_config(
    page_title="Détection Casques EPI",
    page_icon="🪖",
    layout="centered"
)

st.title("🪖 Détection de Casques de Sécurité")
st.write("---")

# Interface simple
st.subheader("📸 Prenez une photo")
img_file = st.camera_input("Cliquez pour prendre une photo")

if img_file is not None:
    # Lire l'image avec PIL (pas besoin de cv2)
    image = Image.open(img_file)
    
    # Afficher l'image
    st.image(image, caption="Photo prise", use_column_width=True)
    
    # Simulation de détection (car YOLO a besoin de cv2)
    st.info("ℹ️ Mode démo - Détection simulée")
    
    # Créer une détection factice
    st.success("✅ 1 personne détectée (simulation)")
    
    # Ajouter un rectangle blanc sur l'image pour simuler la détection
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Dessiner un rectangle blanc (simulation)
    st.image(image, caption="Résultat simulé", use_column_width=True)
    
    st.write("**Note:** Pour la vraie détection YOLO, il faut installer opencv-python-headless")

# Instructions
with st.expander("ℹ️ Informations"):
    st.write("""
    **Problème détecté:** OpenCV (cv2) n'est pas installé.
    
    **Solution:** Vérifiez que votre fichier requirements.txt contient:
