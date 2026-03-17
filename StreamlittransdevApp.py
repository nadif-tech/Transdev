"""
Application de détection de casques de protection avec YOLOv8 et Streamlit
Version simplifiée sans pkg_resources
"""

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
    print("\n📦 Installez les dépendances avec:")
    print("pip install streamlit opencv-python torch ultralytics pillow numpy")
    import sys
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
    
    # Vérification simple des versions
    st.subheader("📦 Versions installées")
    try:
        st.write(f"✅ OpenCV: {cv2.__version__}")
        st.write(f"✅ PyTorch: {torch.__version__}")
        st.write(f"✅ NumPy: {np.__version__}")
        st.write(f"✅ YOLO: {YOLO.__version__}")
    except:
        st.warning("⚠️ Certaines versions ne peuvent pas être affichées")
    
    st.divider()
    
    # Paramètres de détection
    confidence_threshold = st.slider(
        "🎯 Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Plus la valeur est élevée, plus la détection est stricte"
    )
    
    st.divider()
    
    # Instructions
    with st.expander("📖 Instructions", expanded=True):
        st.markdown("""
        **Comment utiliser :**
        1. Choisissez l'onglet **Webcam** ou **Upload Vidéo**
        2. Ajustez le seuil de confiance
        3. Lancez la détection
        
        **Couleurs détectées :**
        - 🔴 Rouge
        - 🟡 Jaune
        - 🔵 Bleu
        - 🟢 Vert
        - ⚪ Blanc
        - 🟠 Orange
        """)

# Chargement du modèle
@st.cache_resource
def load_model():
    """Charge le modèle YOLO"""
    try:
        with st.spinner("🔄 Chargement du modèle YOLO..."):
            model = YOLO('yolov8n.pt')
            st.success("✅ Modèle chargé avec succès!")
            return model
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle: {str(e)}")
        st.info("💡 Essayez de télécharger le modèle avec: from ultralytics import YOLO; YOLO('yolov8n.pt')")
        return None

# Charger le modèle
model = load_model()

if model is None:
    st.stop()

# Classes cibles (0 = personne pour COCO)
TARGET_CLASSES = [0]

# Fonction de détection des couleurs
def detect_helmet_colors(frame, boxes):
    """Détecte la couleur des casques dans les régions détectées"""
    helmet_colors = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Extraire la région de la tête (partie supérieure)
        head_height = (y2 - y1) // 3
        head_region = frame[y1:y1 + head_height, x1:x2]
        
        if head_region.size > 0:
            # Convertir en HSV
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
            
            # Définir les plages de couleurs
            color_ranges = {
                'Rouge': [(0, 100, 100), (10, 255, 255)],
                'Rouge_Fonce': [(160, 100, 100), (180, 255, 255)],
                'Jaune': [(20, 100, 100), (35, 255, 255)],
                'Bleu': [(100, 100, 100), (130, 255, 255)],
                'Vert': [(40, 100, 100), (80, 255, 255)],
                'Blanc': [(0, 0, 200), (180, 30, 255)],
                'Orange': [(10, 100, 100), (20, 255, 255)]
            }
            
            color_found = False
            for color_name, (lower, upper) in color_ranges.items():
                # Créer le masque
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                
                # Combiner les deux masques pour le rouge
                if color_name == 'Rouge':
                    mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
                    mask = cv2.bitwise_or(mask, mask2)
                elif color_name == 'Rouge_Fonce':
                    continue
                
                # Calculer le pourcentage
                color_pixels = cv2.countNonZero(mask)
                total_pixels = head_region.shape[0] * head_region.shape[1]
                
                if total_pixels > 0 and (color_pixels / total_pixels) > 0.15:
                    helmet_colors.append(color_name if color_name != 'Rouge_Fonce' else 'Rouge')
                    color_found = True
                    break
            
            if not color_found:
                helmet_colors.append('Non détecté')
    
    return helmet_colors

# Interface principale
tab1, tab2 = st.tabs(["📹 Webcam en direct", "📁 Analyse vidéo"])

# Onglet Webcam
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📹 Flux Webcam")
        
        # Boutons de contrôle
        col_start, col_stop = st.columns(2)
        with col_start:
            start_webcam = st.button("▶️ Démarrer", type="primary", use_container_width=True)
        with col_stop:
            stop_webcam = st.button("⏹️ Arrêter", use_container_width=True)
        
        if start_webcam and not stop_webcam:
            # Initialisation de la webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Impossible d'accéder à la webcam")
                st.info("💡 Vérifiez que votre webcam est connectée et non utilisée par une autre application")
            else:
                st.success("✅ Webcam activée - Détection en cours...")
                FRAME_WINDOW = st.empty()
                
                # Boucle de capture
                while not stop_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Erreur de lecture")
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
                            color_map = {
                                'Rouge': (0, 0, 255),
                                'Jaune': (0, 255, 255),
                                'Bleu': (255, 0, 0),
                                'Vert': (0, 255, 0),
                                'Blanc': (255, 255, 255),
                                'Orange': (0, 165, 255),
                                'Non détecté': (128, 128, 128)
                            }
                            box_color = color_map.get(color_name, (128, 128, 128))
                            
                            # Dessiner
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(frame, f"Casque: {color_name}", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                    
                    # Ajouter des informations
                    cv2.putText(frame, f"Personnes: {len(boxes)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confiance: {confidence_threshold:.2f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Afficher
                    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    # Petite pause
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cap.release()
    
    with col2:
        st.subheader("📊 Statistiques")
        st.metric("Statut", "Prêt" if not start_webcam else "En cours")
        st.metric("Seuil", f"{confidence_threshold*100:.0f}%")

# Onglet Upload Vidéo
with tab2:
    st.subheader("📁 Analyser une vidéo")
    
    uploaded_file = st.file_uploader(
        "Choisissez un fichier vidéo",
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
            # Obtenir les informations de la vidéo
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps if fps > 0 else 0
            
            st.info(f"📊 Vidéo: {total_frames} frames, {duration:.1f} secondes, {fps} fps")
            
            # Barre de progression
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Affichage vidéo
            stframe = st.empty()
            frame_count = 0
            
            # Traitement
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Traitement: {frame_count}/{total_frames} frames")
                
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
                        
                        color_map = {
                            'Rouge': (0, 0, 255),
                            'Jaune': (0, 255, 255),
                            'Bleu': (255, 0, 0),
                            'Vert': (0, 255, 0),
                            'Blanc': (255, 255, 255),
                            'Orange': (0, 165, 255),
                            'Non détecté': (128, 128, 128)
                        }
                        box_color = color_map.get(color_name, (128, 128, 128))
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(frame, f"Casque: {color_name}", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                
                # Info frame
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            cap.release()
            progress_bar.empty()
            status_text.empty()
            st.success("✅ Analyse terminée!")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🔗 [Documentation YOLOv8](https://docs.ultralytics.com/)")
with col2:
    st.markdown("📚 [Streamlit Docs](https://docs.streamlit.io/)")
with col3:
    st.markdown("🐍 [OpenCV](https://opencv.org/)")

st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.8em;'>"
    "Développé avec ❤️ - Version simplifiée sans dépendances supplémentaires"
    "</p>", 
    unsafe_allow_html=True
)
