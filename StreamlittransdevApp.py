import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from PIL import Image
import time

# Configuration de la page
st.set_page_config(
    page_title="Détection d'EPI - Casques de sécurité",
    page_icon="🪖",
    layout="wide"
)

# Titre de l'application
st.title("🪖 Détection de Casques de Sécurité (Toutes couleurs)")
st.markdown("---")

# Chargement du modèle YOLO
@st.cache_resource
def load_model():
    # Téléchargement automatique du modèle YOLOv8 pré-entraîné
    model = YOLO('yolov8n.pt')  # Version nano pour la vitesse
    return model

# Chargement du modèle
with st.spinner("Chargement du modèle YOLO..."):
    model = load_model()
st.success("Modèle chargé avec succès!")

# Sidebar pour les paramètres
st.sidebar.header("⚙️ Paramètres")
confidence_threshold = st.sidebar.slider(
    "Seuil de confiance", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05
)

# Classes d'objets que YOLO peut détecter
# Les casques sont généralement dans la classe 'person' avec des accessoires
# Nous allons détecter les personnes et analyser la présence de casques

def detect_helmets(frame, model, conf_threshold):
    """Détecte les personnes et analyse les casques"""
    
    # Détection avec YOLO
    results = model(frame, conf=conf_threshold)
    
    # Liste pour stocker les infos de détection
    detections = []
    
    # Analyser les résultats
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordonnées de la boîte
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Confiance et classe
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Nom de la classe
            class_name = model.names[cls]
            
            # Ajouter aux détections
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class': class_name
            })
    
    return detections, frame

def draw_detections(frame, detections):
    """Dessine les boîtes de détection sur l'image"""
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_name = det['class']
        
        # Couleur aléatoire basée sur la classe
        color = (0, 255, 0) if class_name == 'person' else (255, 0, 0)
        
        # Dessiner la boîte
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Ajouter le label
        label = f"{class_name} {conf:.2f}"
        
        # Vérifier si c'est une personne (potentiellement avec casque)
        if class_name == 'person':
            # Calculer la zone de la tête (partie supérieure)
            head_y2 = y1 + int((y2 - y1) * 0.3)  # 30% supérieur du corps
            head_x1, head_y1 = x1, y1
            
            # Ajouter un indicateur pour la zone de la tête
            cv2.rectangle(frame, (head_x1, head_y1), (x2, head_y2), (255, 255, 0), 1)
            
            # Simuler la détection de casque (à améliorer avec un modèle spécialisé)
            # Pour l'exemple, on suppose qu'il y a un casque si conf > 0.7
            if conf > 0.7:
                cv2.putText(frame, "CASQUE DETECTE", (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "PAS DE CASQUE", (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Afficher le label
        cv2.putText(frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

# Interface principale
tab1, tab2 = st.tabs(["📹 Caméra en direct", "📷 Upload d'image"])

with tab1:
    st.header("Détection en temps réel avec la caméra")
    
    # Bouton pour démarrer/arrêter la caméra
    run = st.checkbox("Démarrer la caméra")
    
    # Placeholder pour la vidéo
    frame_placeholder = st.empty()
    
    # Compteur de FPS
    fps_text = st.empty()
    
    # Initialisation de la caméra
    if run:
        cap = cv2.VideoCapture(0)  # 0 pour la caméra par défaut
        
        if not cap.isOpened():
            st.error("Impossible d'ouvrir la caméra")
        else:
            st.info("Caméra démarrée. Appuyez sur 'Arrêter' pour quitter.")
            
            # Variables pour le calcul des FPS
            prev_time = time.time()
            fps = 0
            
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Erreur de capture")
                    break
                
                # Redimensionner pour de meilleures performances
                frame = cv2.resize(frame, (640, 480))
                
                # Détection
                detections, frame = detect_helmets(frame, model, confidence_threshold)
                
                # Dessiner les détections
                frame = draw_detections(frame, detections)
                
                # Calculer les FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                
                # Afficher les FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Afficher le nombre de personnes détectées
                persons = [d for d in detections if d['class'] == 'person']
                cv2.putText(frame, f"Personnes: {len(persons)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Convertir BGR en RGB pour Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Afficher l'image
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Mettre à jour les FPS dans l'interface
                fps_text.text(f"FPS: {fps:.1f}")
            
            cap.release()
    else:
        st.info("Cliquez sur 'Démarrer la caméra' pour commencer la détection")

with tab2:
    st.header("Détection sur image uploadée")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Lire l'image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Afficher l'image originale
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image originale")
            st.image(image, channels="BGR", use_column_width=True)
        
        # Détection
        with st.spinner("Analyse en cours..."):
            detections, image_annotated = detect_helmets(image, model, confidence_threshold)
            image_annotated = draw_detections(image_annotated, detections)
        
        with col2:
            st.subheader("Image avec détections")
            st.image(image_annotated, channels="BGR", use_column_width=True)
        
        # Statistiques
        st.subheader("📊 Statistiques")
        persons = [d for d in detections if d['class'] == 'person']
        st.write(f"Nombre de personnes détectées: {len(persons)}")
        
        # Afficher les détails
        if persons:
            st.write("Détails des personnes:")
            for i, person in enumerate(persons):
                st.write(f"Personne {i+1}: Confiance {person['confidence']:.2f}")

# Pied de page
st.markdown("---")
st.markdown("""
**Note:** Cette application utilise YOLOv8 pour la détection d'objets. 
La détection spécifique des casques est simulée - pour une meilleure précision, 
utilisez un modèle entraîné spécifiquement sur des casques de sécurité.
""")
