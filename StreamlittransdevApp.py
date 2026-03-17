import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import tempfile
from PIL import Image
import numpy as np

# Configuration de la page Streamlit
st.set_page_config(page_title="Détection de Casques", layout="wide")
st.title("🎯 Détection de Casques de Protection en Temps Réel")
st.sidebar.header("⚙️ Paramètres")

# Chargement du modèle YOLOv8
@st.cache_resource
def load_model():
    # Utilisation d'un modèle pré-entraîné (vous pouvez remplacer par votre modèle personnalisé)
    # Pour la détection de casques, vous pouvez utiliser un modèle personnalisé ou yolov8n.pt
    model = YOLO('yolov8n.pt')  # Modèle de base
    # Si vous avez un modèle personnalisé pour les casques, utilisez :
    # model = YOLO('chemin/vers/votre_modele_casque.pt')
    return model

# Initialisation du modèle
try:
    model = load_model()
    st.sidebar.success("✅ Modèle chargé avec succès!")
except Exception as e:
    st.sidebar.error(f"❌ Erreur de chargement du modèle: {e}")

# Options de détection
confidence_threshold = st.sidebar.slider("Seuil de confiance", 0.0, 1.0, 0.5)
use_colors = st.sidebar.checkbox("Détection par couleurs", True)

# Classes d'objets (pour yolov8n.pt, nous filtrons pour les personnes/équipements)
# Si vous utilisez un modèle spécifique aux casques, ajustez ces classes
TARGET_CLASSES = [0]  # 0 = person (dans COCO dataset)
# Pour un modèle casque personnalisé, utilisez les IDs appropriés

# Fonction de détection des couleurs (approche simple)
def detect_helmet_colors(frame, boxes):
    """Détection basique des couleurs de casque"""
    helmet_colors = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        # Extraire la région de la tête (partie supérieure de la boîte)
        head_region = frame[y1:y1 + (y2-y1)//3, x1:x2]
        if head_region.size > 0:
            # Convertir en HSV pour meilleure détection des couleurs
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
            
            # Définir les plages de couleurs pour les casques
            color_ranges = {
                'Rouge': [(0, 70, 50), (10, 255, 255)],
                'Jaune': [(20, 100, 100), (30, 255, 255)],
                'Bleu': [(100, 150, 0), (140, 255, 255)],
                'Vert': [(40, 70, 70), (80, 255, 255)],
                'Blanc': [(0, 0, 200), (180, 30, 255)],
                'Orange': [(10, 100, 100), (20, 255, 255)]
            }
            
            # Détecter la couleur dominante
            for color_name, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                if cv2.countNonZero(mask) > (head_region.size // 10):  # Seuil de 10%
                    helmet_colors.append(color_name)
                    break
            else:
                helmet_colors.append('Inconnu')
    return helmet_colors

# Interface principale
option = st.radio("Choisissez la source:", ["📹 Webcam", "📁 Upload vidéo"])

if option == "📹 Webcam":
    run = st.checkbox("Démarrer la webcam")
    FRAME_WINDOW = st.image([])
    
    cap = None
    if run:
        cap = cv2.VideoCapture(0)
        st.sidebar.info("🎥 Webcam activée - Appuyez sur Stop pour arrêter")
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Erreur de capture webcam")
            break
        
        # Détection avec YOLO
        results = model(frame, conf=confidence_threshold)
        
        # Filtrer pour les classes cibles (personnes)
        boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls in TARGET_CLASSES:  # Ne garder que les personnes
                    boxes.append(box.xyxy[0].tolist())
        
        # Dessiner les boîtes
        if boxes:
            # Détecter les couleurs des casques
            colors = detect_helmet_colors(frame, boxes)
            
            # Annoter l'image
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                color_name = colors[i] if i < len(colors) else "Inconnu"
                
                # Choisir la couleur du rectangle selon le casque
                box_color = {
                    'Rouge': (0, 0, 255),
                    'Jaune': (0, 255, 255),
                    'Bleu': (255, 0, 0),
                    'Vert': (0, 255, 0),
                    'Blanc': (255, 255, 255),
                    'Orange': (0, 165, 255)
                }.get(color_name, (255, 255, 255))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"Casque {color_name}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Afficher le compteur
        cv2.putText(frame, f"Personnes: {len(boxes)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mettre à jour l'affichage
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if cap is not None:
        cap.release()

else:  # Upload vidéo
    uploaded_file = st.file_uploader("Choisissez une vidéo", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Sauvegarder temporairement
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Lire la vidéo
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Détection avec YOLO
            results = model(frame, conf=confidence_threshold)
            
            # Filtrer pour les personnes
            boxes = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls in TARGET_CLASSES:
                        boxes.append(box.xyxy[0].tolist())
            
            # Dessiner les boîtes
            if boxes:
                colors = detect_helmet_colors(frame, boxes)
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box[:4])
                    color_name = colors[i] if i < len(colors) else "Inconnu"
                    
                    box_color = {
                        'Rouge': (0, 0, 255),
                        'Jaune': (0, 255, 255),
                        'Bleu': (255, 0, 0),
                        'Vert': (0, 255, 0),
                        'Blanc': (255, 255, 255),
                        'Orange': (0, 165, 255)
                    }.get(color_name, (255, 255, 255))
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, f"Casque {color_name}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📝 Instructions:
1. Choisissez entre webcam ou upload vidéo
2. Ajustez le seuil de confiance
3. Activez/désactivez la détection par couleurs
4. Lancez la détection

### 🎨 Couleurs détectées:
- Rouge 🔴
- Jaune 🟡
- Bleu 🔵
- Vert 🟢
- Blanc ⚪
- Orange 🟠
""")
