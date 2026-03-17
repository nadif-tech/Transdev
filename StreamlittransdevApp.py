"""
Application de détection de casques de protection avec YOLOv8 et Streamlit
"""

import sys
import subprocess
import pkg_resources

# Vérification et installation automatique des dépendances (optionnel)
required_packages = ['streamlit', 'opencv-python', 'opencv-python-headless', 
                    'torch', 'ultralytics', 'pillow', 'numpy']

def check_and_install_packages():
    """Vérifie et installe les packages manquants"""
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = [pkg for pkg in required_packages if pkg not in installed]
    
    if missing:
        print(f"Packages manquants: {missing}")
        print("Installation en cours...")
        for package in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("Installation terminée. Redémarrez l'application.")
        sys.exit()

# Décommentez la ligne suivante si vous voulez l'installation automatique
# check_and_install_packages()

# Import des modules avec gestion d'erreurs
try:
    import streamlit as st
    import cv2
    import torch
    from ultralytics import YOLO
    import tempfile
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Installez les dépendances avec: pip install streamlit opencv-python torch ultralytics pillow numpy")
    sys.exit(1)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Détection de Casques de Protection",
    page_icon="⛑️",
    layout="wide"
)

# Titre principal avec style
st.markdown("""
    <h1 style='text-align: center; color: #2E86AB;'>
        ⛑️ Détection de Casques de Protection en Temps Réel
    </h1>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Vérification de l'installation
    st.subheader("📦 État des dépendances")
    deps_status = {
        "OpenCV": cv2.__version__ if 'cv2' in dir() else "❌ Non installé",
        "PyTorch": torch.__version__ if 'torch' in dir() else "❌ Non installé",
        "YOLO": "✅ OK" if 'YOLO' in dir() else "❌ Non installé",
        "NumPy": np.__version__ if 'np' in dir() else "❌ Non installé"
    }
    
    for dep, version in deps_status.items():
        st.write(f"{dep}: {version}")
    
    st.divider()
    
    # Paramètres de détection
    confidence_threshold = st.slider(
        "🎯 Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Plus la valeur est élevée, plus la détection est stricte"
    )
    
    # Option pour charger un modèle personnalisé
    use_custom_model = st.checkbox("Utiliser un modèle personnalisé", False)
    
    if use_custom_model:
        uploaded_model = st.file_uploader(
            "Chargez votre modèle YOLO",
            type=['pt'],
            help="Fichier .pt de votre modèle entraîné"
        )
    
    st.divider()
    
    # Instructions
    with st.expander("📖 Instructions"):
        st.markdown("""
        1. **Installation** : `pip install -r requirements.txt`
        2. **Lancement** : `streamlit run app.py`
        3. **Choisissez** : Webcam ou upload vidéo
        4. **Ajustez** le seuil de confiance
        5. **Détection** automatique des casques
        
        **Couleurs supportées :**
        - 🔴 Rouge
        - 🟡 Jaune
        - 🔵 Bleu
        - 🟢 Vert
        - ⚪ Blanc
        - 🟠 Orange
        """)

# Chargement du modèle avec gestion d'erreur
@st.cache_resource
def load_model(model_path=None):
    """Charge le modèle YOLO"""
    try:
        if model_path and model_path is not None:
            # Si un modèle personnalisé est fourni
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                tmp_file.write(model_path.read())
                model = YOLO(tmp_file.name)
        else:
            # Modèle par défaut
            model = YOLO('yolov8n.pt')
        
        # Classes d'objets (ajustez selon votre modèle)
        model.class_names = ['personne']  # Pour le modèle de base
        return model
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle: {str(e)}")
        return None

# Charger le modèle
if use_custom_model and 'uploaded_model' in locals() and uploaded_model:
    model = load_model(uploaded_model)
else:
    model = load_model()

if model is None:
    st.stop()

# Classes cibles (à modifier selon votre modèle)
TARGET_CLASSES = [0]  # 0 = personne pour COCO dataset

# Fonction de détection des couleurs
def detect_helmet_colors(frame, boxes):
    """Détecte la couleur des casques dans les régions détectées"""
    helmet_colors = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Extraire la région de la tête (partie supérieure de la boîte)
        head_height = (y2 - y1) // 3
        head_region = frame[y1:y1 + head_height, x1:x2]
        
        if head_region.size > 0:
            # Convertir en HSV
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
            
            # Définir les plages de couleurs
            color_ranges = {
                'Rouge': [(0, 100, 100), (10, 255, 255)],
                'Rouge_2': [(160, 100, 100), (180, 255, 255)],  # Pour les rouges foncés
                'Jaune': [(20, 100, 100), (35, 255, 255)],
                'Bleu': [(100, 100, 100), (130, 255, 255)],
                'Vert': [(40, 100, 100), (80, 255, 255)],
                'Blanc': [(0, 0, 200), (180, 30, 255)],
                'Orange': [(10, 100, 100), (20, 255, 255)]
            }
            
            color_found = False
            for color_name, (lower, upper) in color_ranges.items():
                if color_name == 'Rouge_2':
                    continue  # On traite les deux plages rouge ensemble
                
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                
                # Pour le rouge, ajouter la deuxième plage
                if color_name == 'Rouge':
                    mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
                    mask = cv2.bitwise_or(mask, mask2)
                
                # Calculer le pourcentage de pixels de couleur
                color_pixels = cv2.countNonZero(mask)
                total_pixels = head_region.shape[0] * head_region.shape[1]
                
                if total_pixels > 0 and (color_pixels / total_pixels) > 0.15:  # Seuil de 15%
                    helmet_colors.append(color_name)
                    color_found = True
                    break
            
            if not color_found:
                helmet_colors.append('Non détecté')
    
    return helmet_colors

# Interface principale
tab1, tab2 = st.tabs(["📹 Webcam", "📁 Upload Vidéo"])

# Onglet Webcam
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Flux Webcam en Direct")
        run_webcam = st.button("▶️ Démarrer la webcam", type="primary")
        stop_webcam = st.button("⏹️ Arrêter")
        
        if run_webcam and not stop_webcam:
            FRAME_WINDOW = st.empty()
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Impossible d'accéder à la webcam")
            else:
                st.success("✅ Webcam activée - Détection en cours...")
                
                while not stop_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Erreur de lecture de la webcam")
                        break
                    
                    # Détection YOLO
                    results = model(frame, conf=confidence_threshold)
                    
                    # Extraire les boîtes
                    boxes = []
                    if results[0].boxes is not None:
                        for box in results[0].boxes:
                            cls = int(box.cls[0])
                            if cls in TARGET_CLASSES:
                                boxes.append(box.xyxy[0].tolist())
                    
                    # Détection des couleurs
                    if boxes:
                        colors = detect_helmet_colors(frame, boxes)
                        
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box[:4])
                            color_name = colors[i] if i < len(colors) else "Inconnu"
                            
                            # Couleur du rectangle
                            box_color = {
                                'Rouge': (0, 0, 255),
                                'Jaune': (0, 255, 255),
                                'Bleu': (255, 0, 0),
                                'Vert': (0, 255, 0),
                                'Blanc': (255, 255, 255),
                                'Orange': (0, 165, 255)
                            }.get(color_name, (128, 128, 128))
                            
                            # Dessiner
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(frame, f"Casque: {color_name}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    
                    # Info
                    cv2.putText(frame, f"Personnes: {len(boxes)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Afficher
                    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    # Petite pause pour éviter la surcharge
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cap.release()
    
    with col2:
        st.subheader("📊 Statistiques")
        st.metric("Statut", "Prêt")
        st.metric("Confiance", f"{confidence_threshold*100:.0f}%")

# Onglet Upload Vidéo
with tab2:
    uploaded_file = st.file_uploader(
        "Choisissez une vidéo",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Formats supportés: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file is not None:
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Lire la vidéo
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("❌ Erreur d'ouverture de la vidéo")
        else:
            stframe = st.empty()
            progress_bar = st.progress(0)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)
                
                # Détection YOLO
                results = model(frame, conf=confidence_threshold)
                
                # Traitement identique à la webcam
                boxes = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        if cls in TARGET_CLASSES:
                            boxes.append(box.xyxy[0].tolist())
                
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
                        }.get(color_name, (128, 128, 128))
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(frame, f"Casque: {color_name}", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            cap.release()
            progress_bar.empty()
            st.success("✅ Traitement terminé!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Développé avec ❤️ utilisant YOLOv8 et Streamlit"
    "</p>", 
    unsafe_allow_html=True
)
