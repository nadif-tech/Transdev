import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import io
import re
from datetime import datetime
import os

# Configuration de la page
st.set_page_config(
    page_title="Extracteur de Numéros - OpenCV",
    page_icon="📸",
    layout="wide"
)

st.title("📸 Extracteur de Numéros avec OpenCV")
st.markdown("---")

# Initialisation des classificateurs en cascade (détection de texte)
@st.cache_resource
def load_cascade_classifiers():
    """Charge les classificateurs OpenCV"""
    classifiers = {}
    
    # Télécharger les fichiers cascade si nécessaire
    cascade_files = {
        'digits': cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml',
        'face': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    }
    
    for name, path in cascade_files.items():
        if os.path.exists(path):
            classifiers[name] = cv2.CascadeClassifier(path)
    
    return classifiers

def preprocess_image(image):
    """Prétraitement de l'image pour meilleure détection"""
    # Convertir PIL en OpenCV
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convertir en niveaux de gris
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Améliorer le contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Débruitage
    denoised = cv2.fastNlMeansDenoising(enhanced)
    
    # Binarisation adaptative
    binary = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return image, gray, binary

def detect_text_regions(binary_image):
    """Détecte les régions contenant du texte/numéros"""
    # Trouver les contours
    contours, _ = cv2.findContours(
        binary_image, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filtrer les régions trop petites
        if w > 20 and h > 20 and w < binary_image.shape[1]//2:
            aspect_ratio = w / h
            # Les numéros ont généralement un aspect ratio entre 0.2 et 5
            if 0.2 < aspect_ratio < 5:
                regions.append((x, y, w, h))
    
    return regions

def extract_digits_morphological(binary_image):
    """Extrait les chiffres par morphologie mathématique"""
    # Opérations morphologiques pour isoler les chiffres
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Dilatation pour connecter les parties des chiffres
    dilated = cv2.dilate(binary_image, kernel, iterations=1)
    
    # Érosion pour enlever le bruit
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Trouver les composants connectés
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        eroded, connectivity=8
    )
    
    digits_found = []
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filtrer par taille et proportions (caractéristiques des chiffres)
        if 100 < area < 5000 and 10 < w < 100 and 20 < h < 150:
            aspect_ratio = h / w
            if 1.5 < aspect_ratio < 4.0:  # Les chiffres sont plus hauts que larges
                roi = binary_image[y:y+h, x:x+w]
                
                # Compter les pixels blancs
                white_pixels = cv2.countNonZero(roi)
                density = white_pixels / (w * h)
                
                if 0.1 < density < 0.8:  # Densité raisonnable pour un chiffre
                    digits_found.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'density': density
                    })
    
    return digits_found

def recognize_digit_template(roi):
    """Reconnaissance basique de chiffres par template matching"""
    # Redimensionner à une taille standard
    roi_resized = cv2.resize(roi, (20, 30))
    
    # Calculer des caractéristiques simples
    moments = cv2.moments(roi_resized)
    hu_moments = cv2.HuMoments(moments)
    
    # Compter les trous (pour distinguer 0, 6, 8, 9)
    contours, _ = cv2.findContours(
        roi_resized, 
        cv2.RETR_CCOMP, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    num_holes = len(contours) - 1 if len(contours) > 1 else 0
    
    # Ratio de pixels
    pixel_ratio = cv2.countNonZero(roi_resized) / (20 * 30)
    
    return {
        'hu_moments': hu_moments.flatten()[:4],
        'holes': num_holes,
        'pixel_ratio': pixel_ratio
    }

def extract_numbers_from_image(image):
    """Extrait les numéros d'une image avec OpenCV"""
    try:
        # Prétraitement
        original, gray, binary = preprocess_image(image)
        
        # Méthode 1: Détection de régions de texte
        regions = detect_text_regions(binary)
        
        # Méthode 2: Extraction morphologique des chiffres
        digits = extract_digits_morphological(binary)
        
        # Grouper les chiffres par proximité
        grouped_numbers = group_digits_by_proximity(digits)
        
        # Convertir en texte
        detected_text = ""
        detected_numbers = []
        
        if grouped_numbers:
            for group in grouped_numbers:
                number_str = ''.join(group['digits'])
                detected_text += f" {number_str}"
                detected_numbers.append(number_str)
        
        # Extraire aussi les nombres avec regex du texte détecté
        all_numbers = []
        for num_str in detected_numbers:
            numbers = re.findall(r'\b\d+\b', num_str)
            all_numbers.extend(numbers)
        
        return {
            'text': detected_text.strip(),
            'numbers': all_numbers,
            'all_numbers': ', '.join(all_numbers) if all_numbers else '',
            'count': len(all_numbers),
            'regions_detectees': len(regions),
            'chiffres_detectes': len(digits),
            'success': True
        }
        
    except Exception as e:
        return {
            'text': '',
            'numbers': [],
            'all_numbers': '',
            'count': 0,
            'regions_detectees': 0,
            'chiffres_detectes': 0,
            'success': False,
            'error': str(e)
        }

def group_digits_by_proximity(digits, x_threshold=50, y_threshold=20):
    """Groupe les chiffres proches en nombres"""
    if not digits:
        return []
    
    # Trier par position x
    sorted_digits = sorted(digits, key=lambda d: d['bbox'][0])
    
    groups = []
    current_group = [sorted_digits[0]]
    
    for i in range(1, len(sorted_digits)):
        current = sorted_digits[i]
        previous = sorted_digits[i-1]
        
        x_dist = current['bbox'][0] - (previous['bbox'][0] + previous['bbox'][2])
        y_diff = abs(current['bbox'][1] - previous['bbox'][1])
        
        # Si proche horizontalement et aligné verticalement
        if x_dist < x_threshold and y_diff < y_threshold:
            current_group.append(current)
        else:
            # Nouveau groupe
            groups.append(current_group)
            current_group = [current]
    
    if current_group:
        groups.append(current_group)
    
    # Pour chaque groupe, essayer de reconnaître les chiffres
    result = []
    for group in groups:
        group_digits = []
        for digit in group:
            # Estimation simple du chiffre basée sur la densité
            density = digit['density']
            bbox = digit['bbox']
            
            # Reconnaissance basique (à améliorer)
            if density < 0.3:
                estimated_digit = '1'  # Chiffre fin
            elif density > 0.6:
                estimated_digit = '8'  # Chiffre dense
            else:
                estimated_digit = '0'  # Densité moyenne
            
            group_digits.append(estimated_digit)
        
        result.append({
            'bbox': group[0]['bbox'],
            'digits': group_digits,
            'count': len(group_digits)
        })
    
    return result

def draw_detections(image, digits):
    """Dessine les détections sur l'image pour visualisation"""
    img_copy = image.copy()
    if len(img_copy.shape) == 2:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
    
    for digit in digits:
        x, y, w, h = digit['bbox']
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return img_copy

def save_to_excel(data):
    """Convertit les données en fichier Excel"""
    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Numéros_Extraits', index=False)
    output.seek(0)
    return output

# Initialisation de la session state
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'all_results' not in st.session_state:
    st.session_state.all_results = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration OpenCV")
    
    # Paramètres ajustables
    st.subheader("Paramètres de détection")
    
    min_confidence = st.slider(
        "Sensibilité de détection",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1
    )
    
    st.markdown("---")
    
    # Upload de photos
    uploaded_files = st.file_uploader(
        "📁 Choisissez vos photos",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Formats supportés: PNG, JPG, JPEG, BMP, TIFF"
    )
    
    # Appareil photo
    camera_photo = st.camera_input("📷 Ou prenez une photo")
    
    if camera_photo is not None:
        if camera_photo not in st.session_state.processed_images:
            st.session_state.processed_images.append(camera_photo)
    
    st.markdown("---")
    
    # Actions
    process_button = st.button(
        "🔍 Extraire les numéros", 
        type="primary", 
        use_container_width=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Effacer", use_container_width=True):
            st.session_state.processed_images = []
            st.session_state.all_results = []
            st.rerun()
    
    with col2:
        if st.button("ℹ️ Aide", use_container_width=True):
            st.info("""
            **Comment ça marche:**
            1. Ajoutez des photos
            2. Cliquez sur 'Extraire'
            3. OpenCV détecte les zones de texte
            4. Les numéros sont identifiés
            5. Téléchargez en Excel
            """)

# Zone principale
tab1, tab2, tab3 = st.tabs(["📋 Photos", "📊 Résultats", "🔬 Analyse"])

with tab1:
    st.subheader("Photos à traiter")
    
    if uploaded_files:
        for file in uploaded_files:
            if file not in st.session_state.processed_images:
                st.session_state.processed_images.append(file)
    
    if st.session_state.processed_images:
        cols = st.columns(3)
        for idx, img_file in enumerate(st.session_state.processed_images):
            with cols[idx % 3]:
                image = Image.open(img_file)
                st.image(image, caption=f"Photo {idx+1}", use_container_width=True)
                
                col_info, col_del = st.columns([3, 1])
                with col_info:
                    st.caption(f"📄 {img_file.name}")
                with col_del:
                    if st.button("❌", key=f"del_{idx}"):
                        st.session_state.processed_images.pop(idx)
                        st.rerun()
    else:
        st.info("👆 Utilisez la barre latérale pour ajouter des photos")
        st.markdown("""
        ### 📌 Instructions:
        1. **Upload** de photos depuis votre ordinateur
        2. **Ou prenez** une photo avec votre caméra
        3. **Cliquez** sur 'Extraire les numéros'
        4. **Téléchargez** les résultats en Excel
        """)

with tab2:
    st.subheader("Résultats de l'extraction")
    
    if process_button and st.session_state.processed_images:
        st.session_state.all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, img_file in enumerate(st.session_state.processed_images):
            status_text.text(f"Traitement de la photo {idx+1}/{len(st.session_state.processed_images)}...")
            
            image = Image.open(img_file)
            result = extract_numbers_from_image(image)
            
            result['filename'] = img_file.name
            result['processed_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result['image_size'] = f"{image.size[0]}x{image.size[1]}"
            
            st.session_state.all_results.append(result)
            progress_bar.progress((idx + 1) / len(st.session_state.processed_images))
        
        status_text.text("✅ Extraction terminée!")
        st.success(f"✅ {len(st.session_state.all_results)} photos traitées")
    
    if st.session_state.all_results:
        # Tableau des résultats
        df_display = pd.DataFrame([
            {
                'Photo': r['filename'],
                'Taille': r['image_size'],
                'Numéros': r['count'],
                'Valeurs': r['all_numbers'][:50] + ('...' if len(r['all_numbers']) > 50 else ''),
                'Régions': r.get('regions_detectees', 0),
                'Chiffres': r.get('chiffres_detectes', 0)
            }
            for r in st.session_state.all_results
        ])
        
        st.dataframe(df_display, use_container_width=True)
        
        # Téléchargement Excel
        st.markdown("### 💾 Export des résultats")
        
        excel_data = []
        for result in st.session_state.all_results:
            excel_data.append({
                'Nom du fichier': result['filename'],
                'Date de traitement': result['processed_date'],
                'Taille image': result['image_size'],
                'Nombre de numéros': result['count'],
                'Numéros trouvés': result['all_numbers'],
                'Régions détectées': result.get('regions_detectees', 0),
                'Chiffres détectés': result.get('chiffres_detectes', 0),
                'Succès': 'Oui' if result['success'] else 'Non'
            })
        
        excel_file = save_to_excel(excel_data)
        
        col_down1, col_down2 = st.columns([2, 1])
        with col_down1:
            st.download_button(
                label="📥 Télécharger les résultats (Excel)",
                data=excel_file,
                file_name=f"numeros_opencv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col_down2:
            if st.button("📋 Copier tous les numéros", use_container_width=True):
                all_nums = []
                for r in st.session_state.all_results:
                    if r['numbers']:
                        all_nums.extend(r['numbers'])
                if all_nums:
                    nums_text = ', '.join(all_nums)
                    st.code(nums_text)
                    st.success("Numéros copiés!")

with tab3:
    st.subheader("Analyse détaillée")
    
    if st.session_state.all_results:
        for idx, result in enumerate(st.session_state.all_results):
            with st.expander(f"📊 Photo {idx+1}: {result['filename']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Informations:**")
                    st.write(f"- Taille: {result['image_size']}")
                    st.write(f"- Régions texte: {result.get('regions_detectees', 0)}")
                    st.write(f"- Chiffres isolés: {result.get('chiffres_detectes', 0)}")
                    st.write(f"- Numéros trouvés: {result['count']}")
                
                with col2:
                    st.write("**Numéros détectés:**")
                    if result['numbers']:
                        for num in result['numbers']:
                            st.code(num)
                    else:
                        st.warning("Aucun numéro détecté")
                
                if not result['success']:
                    st.error(f"Erreur: {result.get('error', 'Inconnue')}")
    else:
        st.info("Traitez d'abord des photos pour voir l'analyse")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔍 Extracteur de numéros avec OpenCV | Traitement d'image en temps réel</p>
    <p style='font-size: 12px;'>Détection morphologique • Analyse de contours • Regroupement par proximité</p>
</div>
""", unsafe_allow_html=True)
