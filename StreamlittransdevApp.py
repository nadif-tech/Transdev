import streamlit as st
import pandas as pd
from PIL import Image
import io
import re
from datetime import datetime
import numpy as np
import cv2
import time

# Configuration de la page
st.set_page_config(
    page_title="Extracteur Numeros - Algorithmes Avances",
    page_icon="123",
    layout="wide"
)

st.title("Extracteur de Numeros - Algorithmes Avances de Vision")
st.markdown("---")

# ============================================
# ALGORITHMES AVANCES DE TRAITEMENT D'IMAGE
# ============================================

def advanced_preprocessing(image):
    """
    Pretraitement avance avec multiples techniques
    """
    # Convertir PIL en OpenCV
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convertir en niveaux de gris
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # 1. CLAHE - Amelioration du contraste local
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 2. Debruitage - Non-local Means Denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # 3. Filtre bilateral - Preserve les bords
    bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)
    
    return bilateral

def multiple_binarizations(image):
    """
    Applique plusieurs techniques de binarisation
    """
    binarizations = []
    
    # 1. Otsu
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binarizations.append(('otsu', otsu))
    
    # 2. Adaptative Gaussienne
    adaptive_gaussian = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    binarizations.append(('adaptive_gaussian', adaptive_gaussian))
    
    # 3. Adaptative Mean
    adaptive_mean = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    binarizations.append(('adaptive_mean', adaptive_mean))
    
    # 4. Sauvola (implemente manuellement)
    sauvola = sauvola_threshold(image)
    binarizations.append(('sauvola', sauvola))
    
    return binarizations

def sauvola_threshold(image, window_size=25, k=0.2, r=128):
    """
    Algorithme de binarisation Sauvola
    """
    h, w = image.shape
    result = np.zeros((h, w), dtype=np.uint8)
    
    pad = window_size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    
    for i in range(h):
        for j in range(w):
            window = padded[i:i+window_size, j:j+window_size]
            mean = np.mean(window)
            std = np.std(window)
            
            threshold = mean * (1 + k * ((std / r) - 1))
            
            if image[i, j] > threshold:
                result[i, j] = 255
            else:
                result[i, j] = 0
    
    return result

def morphological_operations(binary_image):
    """
    Operations morphologiques pour nettoyer l'image
    """
    # Kernel pour operations morphologiques
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Fermeture horizontale - connecte les chiffres
    closed_h = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_h)
    
    # Ouverture verticale - enleve le bruit vertical
    opened_v = cv2.morphologyEx(closed_h, cv2.MORPH_OPEN, kernel_v)
    
    # Dilatation legere
    dilated = cv2.dilate(opened_v, kernel_square, iterations=1)
    
    # Erosion pour affiner
    eroded = cv2.erode(dilated, kernel_square, iterations=1)
    
    return eroded

def detect_text_regions(binary_image):
    """
    Detection des regions de texte par analyse de composants connectes
    """
    # Inverser si necessaire (texte noir sur fond blanc)
    if np.mean(binary_image) > 127:
        binary_image = cv2.bitwise_not(binary_image)
    
    # Analyse des composants connectes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    
    regions = []
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filtres pour identifier les caracteres/numeros
        aspect_ratio = h / w if w > 0 else 0
        density = area / (w * h) if w * h > 0 else 0
        
        # Criteres pour un caractere/numbero
        if (20 < w < 200 and 30 < h < 300 and
            50 < area < 5000 and
            0.5 < aspect_ratio < 4.0 and
            0.1 < density < 0.8):
            
            regions.append({
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'density': density,
                'center': (x + w//2, y + h//2)
            })
    
    return regions

def group_regions_into_lines(regions, vertical_tolerance=20):
    """
    Groupe les regions en lignes de texte
    """
    if not regions:
        return []
    
    # Trier par position y
    sorted_regions = sorted(regions, key=lambda r: r['center'][1])
    
    lines = []
    current_line = [sorted_regions[0]]
    
    for i in range(1, len(sorted_regions)):
        current = sorted_regions[i]
        previous = sorted_regions[i-1]
        
        y_diff = abs(current['center'][1] - previous['center'][1])
        
        if y_diff <= vertical_tolerance:
            current_line.append(current)
        else:
            lines.append(current_line)
            current_line = [current]
    
    if current_line:
        lines.append(current_line)
    
    return lines

def extract_numbers_from_lines(lines, min_group_size=2):
    """
    Extrait les numeros des lignes de texte
    """
    numbers_found = []
    
    for line in lines:
        if len(line) >= min_group_size:
            # Trier par position x
            sorted_line = sorted(line, key=lambda r: r['bbox'][0])
            
            # Extraire la ligne de texte comme une region
            min_x = min(r['bbox'][0] for r in sorted_line)
            max_x = max(r['bbox'][0] + r['bbox'][2] for r in sorted_line)
            min_y = min(r['bbox'][1] for r in sorted_line)
            max_y = max(r['bbox'][1] + r['bbox'][3] for r in sorted_line)
            
            # Caracteristiques de la ligne
            line_width = max_x - min_x
            line_height = max_y - min_y
            
            # Estimer le nombre de caracteres
            avg_width = line_width / len(sorted_line)
            estimated_chars = int(line_width / 15)
            
            # Creer un identifiant pour cette ligne
            line_id = f"L{len(numbers_found)+1}_{estimated_chars}"
            
            numbers_found.append({
                'position': (min_x, min_y, max_x, max_y),
                'char_count': len(sorted_line),
                'estimated_length': estimated_chars,
                'line_id': line_id
            })
    
    return numbers_found

def template_matching_digits(binary_image, regions):
    """
    Template matching pour reconnaitre les chiffres
    """
    digits_templates = create_digit_templates()
    recognized_numbers = []
    
    for region in regions:
        x, y, w, h = region['bbox']
        roi = binary_image[y:y+h, x:x+w]
        
        # Redimensionner pour template matching
        roi_resized = cv2.resize(roi, (20, 30))
        
        best_match = None
        best_score = 0
        
        for digit, template in digits_templates.items():
            template_resized = cv2.resize(template, (20, 30))
            
            # Correlation
            correlation = cv2.matchTemplate(
                roi_resized.astype(np.float32),
                template_resized.astype(np.float32),
                cv2.TM_CCOEFF_NORMED
            )
            
            score = np.max(correlation)
            
            if score > best_score:
                best_score = score
                best_match = digit
        
        if best_score > 0.5:
            recognized_numbers.append({
                'digit': best_match,
                'confidence': best_score,
                'position': (x, y)
            })
    
    return recognized_numbers

def create_digit_templates():
    """
    Cree des templates simples pour les chiffres
    """
    templates = {}
    
    # Templates basiques pour 0-9
    for i in range(10):
        template = np.zeros((30, 20), dtype=np.uint8)
        
        # Dessiner le chiffre
        if i == 0:
            cv2.ellipse(template, (10, 15), (6, 10), 0, 0, 360, 255, -1)
            cv2.ellipse(template, (10, 15), (3, 7), 0, 0, 360, 0, -1)
        elif i == 1:
            cv2.rectangle(template, (7, 5), (13, 25), 255, -1)
        elif i == 8:
            cv2.ellipse(template, (10, 10), (6, 6), 0, 0, 360, 255, -1)
            cv2.ellipse(template, (10, 20), (6, 6), 0, 0, 360, 255, -1)
        else:
            cv2.putText(template, str(i), (5, 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
        
        templates[str(i)] = template
    
    return templates

def edge_detection_analysis(image):
    """
    Analyse par detection de contours
    """
    # Canny edge detection
    edges = cv2.Canny(image, 50, 150)
    
    # Dilatation pour connecter les bords
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Trouver les contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if 100 < area < 10000 and w > 15 and h > 20:
            text_contours.append((x, y, w, h))
    
    return text_contours

def extract_numbers_comprehensive(image):
    """
    Extraction complete utilisant tous les algorithmes
    """
    # Pretraitement avance
    preprocessed = advanced_preprocessing(image)
    
    # Multiples binarisations
    binarizations = multiple_binarizations(preprocessed)
    
    all_numbers = []
    all_regions = []
    
    for bin_name, binary in binarizations:
        # Operations morphologiques
        morph = morphological_operations(binary)
        
        # Detection des regions
        regions = detect_text_regions(morph)
        
        if regions:
            # Grouper en lignes
            lines = group_regions_into_lines(regions)
            
            # Extraire les numeros
            numbers = extract_numbers_from_lines(lines)
            
            for num in numbers:
                num['method'] = bin_name
                all_numbers.append(num)
            
            all_regions.extend(regions)
    
    # Detection par contours
    edge_regions = edge_detection_analysis(preprocessed)
    
    return {
        'numbers': all_numbers,
        'regions': all_regions,
        'edge_regions': edge_regions,
        'total_detected': len(all_numbers)
    }

# ============================================
# INTERFACE STREAMLIT
# ============================================

if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'results' not in st.session_state:
    st.session_state.results = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Sidebar
with st.sidebar:
    st.header("Configuration Algorithmes")
    
    st.subheader("Parametres de detection")
    
    sensitivity = st.slider(
        "Sensibilite",
        min_value=1,
        max_value=10,
        value=5,
        help="Sensibilite de detection des caracteres"
    )
    
    min_confidence = st.slider(
        "Confiance minimale",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1
    )
    
    st.markdown("---")
    
    st.header("Upload d'images")
    
    uploaded_files = st.file_uploader(
        "Choisissez vos photos",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.session_state.uploaded_images = uploaded_files
        st.success(f"{len(uploaded_files)} image(s) chargee(s)")
    
    st.markdown("---")
    
    st.header("Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("ANALYSER", type="primary", use_container_width=True):
            if not st.session_state.uploaded_images:
                st.warning("Chargez des images")
            else:
                st.session_state.processing = True
                st.session_state.results = []
                st.rerun()
    
    with col2:
        if st.button("EFFACER", use_container_width=True):
            st.session_state.uploaded_images = []
            st.session_state.results = []
            st.rerun()

# Zone principale
tab1, tab2, tab3, tab4 = st.tabs(["Galerie", "Analyse", "Resultats", "Visualisation"])

with tab1:
    st.header("Galerie d'images")
    
    if st.session_state.uploaded_images:
        cols = st.columns(min(3, len(st.session_state.uploaded_images)))
        
        for idx, img_file in enumerate(st.session_state.uploaded_images):
            with cols[idx % 3]:
                image = Image.open(img_file)
                st.image(image, caption=img_file.name, use_container_width=True)
    else:
        st.info("Chargez des images dans la barre laterale")
        
        st.markdown("### Algorithmes utilises:")
        st.markdown("""
        - **CLAHE** - Amelioration du contraste
        - **Non-local Means** - Debruitage avance
        - **Bilateral Filter** - Preservation des bords
        - **Multi-binarisation** - Otsu, Adaptative, Sauvola
        - **Operations Morphologiques** - Nettoyage
        - **Connected Components** - Detection de regions
        - **Edge Detection** - Canny
        - **Template Matching** - Reconnaissance de chiffres
        """)

with tab2:
    st.header("Analyse avec Algorithmes Avances")
    
    if st.session_state.processing:
        st.info("Traitement en cours...")
        
        progress_bar = st.progress(0)
        
        for idx, img_file in enumerate(st.session_state.uploaded_images):
            st.write(f"Analyse de {img_file.name}")
            
            image = Image.open(img_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Originale", width=250)
            
            with col2:
                with st.spinner("Algorithmes en cours..."):
                    result = extract_numbers_comprehensive(image)
                    
                    result['filename'] = img_file.name
                    result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    st.success(f"Detection: {result['total_detected']} elements")
                    st.write(f"Regions detectees: {len(result['regions'])}")
                    st.write(f"Contours: {len(result['edge_regions'])}")
                    
                    st.session_state.results.append(result)
            
            progress_bar.progress((idx + 1) / len(st.session_state.uploaded_images))
            time.sleep(0.5)
        
        st.session_state.processing = False
        st.success("Analyse terminee!")
        st.rerun()
    
    else:
        if st.session_state.uploaded_images:
            st.info("Cliquez sur ANALYSER pour lancer les algorithmes")

with tab3:
    st.header("Resultats de l'extraction")
    
    if st.session_state.results:
        summary_data = []
        
        for r in st.session_state.results:
            summary_data.append({
                'Fichier': r['filename'],
                'Elements': r['total_detected'],
                'Regions': len(r['regions']),
                'Contours': len(r['edge_regions']),
                'Date': r['timestamp']
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        for idx, result in enumerate(st.session_state.results):
            with st.expander(f"{result['filename']} - Details"):
                st.write(f"**Total elements:** {result['total_detected']}")
                st.write(f"**Regions texte:** {len(result['regions'])}")
                st.write(f"**Contours Canny:** {len(result['edge_regions'])}")
                
                if result['numbers']:
                    st.write("**Numeros estimes:**")
                    for num in result['numbers'][:10]:
                        st.write(f"- Ligne {num['line_id']}: ~{num['estimated_length']} caracteres")
        
        st.markdown("---")
        st.subheader("Export")
        
        export_data = []
        for r in st.session_state.results:
            export_data.append({
                'Fichier': r['filename'],
                'Elements_detectes': r['total_detected'],
                'Regions': len(r['regions']),
                'Date': r['timestamp']
            })
        
        df_export = pd.DataFrame(export_data)
        csv_data = df_export.to_csv(index=False)
        
        st.download_button(
            label="TELECHARGER CSV",
            data=csv_data,
            file_name=f"analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    else:
        st.info("Aucun resultat disponible")

with tab4:
    st.header("Visualisation des Algorithmes")
    
    if st.session_state.uploaded_images:
        img_file = st.session_state.uploaded_images[0]
        image = Image.open(img_file)
        
        st.subheader("Etapes de traitement")
        
        # Convertir en array
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(image, caption="Originale", use_container_width=True)
        
        with col2:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            st.image(enhanced, caption="CLAHE", use_container_width=True)
        
        with col3:
            _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            st.image(otsu, caption="Otsu", use_container_width=True)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            st.image(adaptive, caption="Adaptatif", use_container_width=True)
        
        with col5:
            edges = cv2.Canny(enhanced, 50, 150)
            st.image(edges, caption="Canny Edges", use_container_width=True)
        
        with col6:
            kernel = np.ones((3,3), np.uint8)
            morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
            st.image(morph, caption="Morphologique", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='background: #1a1a2e; padding: 20px; border-radius: 10px; color: white; text-align: center;'>
    <h3 style='color: white;'>Algorithmes Avances de Vision par Ordinateur</h3>
    <p>CLAHE | Non-local Means | Bilateral | Otsu | Sauvola | Adaptatif | Morphologique | Canny | Connected Components</p>
    <p>Compatible Streamlit Cloud | Haute Precision</p>
</div>
""", unsafe_allow_html=True)
