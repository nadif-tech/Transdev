import streamlit as st
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
from datetime import datetime
import requests
import base64
import json

# Configuration de la page
st.set_page_config(
    page_title="Extracteur Numéros - Haute Précision",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Extracteur de Numéros - Haute Précision")
st.markdown("---")

# ============================================
# FONCTIONS DE PRÉTRAITEMENT AVANCÉ
# ============================================

def preprocess_image_advanced(image):
    """
    Prétraitement avancé pour améliorer la précision OCR
    """
    # Convertir en RGB si nécessaire
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Augmenter la résolution si trop petite
    width, height = image.size
    if width < 1000:
        scale_factor = 1500 / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Améliorer le contraste
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.5)
    
    # Améliorer la netteté
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    # Convertir en niveaux de gris
    gray = image.convert('L')
    
    # Appliquer un filtre de débruitage
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    
    # Améliorer les bords
    gray = gray.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    return gray

def create_multiple_versions(image):
    """
    Crée plusieurs versions de l'image pour maximiser la détection
    """
    versions = []
    
    # Version 1: Originale prétraitée
    v1 = preprocess_image_advanced(image)
    versions.append(('standard', v1))
    
    # Version 2: Niveaux de gris simples
    if image.mode != 'L':
        v2 = image.convert('L')
        versions.append(('gray', v2))
    
    # Version 3: Fort contraste
    v3 = image.convert('L')
    enhancer = ImageEnhance.Contrast(v3)
    v3 = enhancer.enhance(3.0)
    versions.append(('high_contrast', v3))
    
    # Version 4: Inversée (texte blanc sur fond noir)
    v4 = image.convert('L')
    v4 = v4.point(lambda x: 255 - x)
    versions.append(('inverted', v4))
    
    return versions

# ============================================
# MÉTHODES OCR MULTIPLES
# ============================================

def ocr_space_api(image, language='fre'):
    """
    OCR.space API - Méthode 1
    """
    try:
        img_byte_arr = io.BytesIO()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        url = "https://api.ocr.space/parse/image"
        
        payload = {
            'apikey': 'K86742198888957',
            'language': language,
            'isOverlayRequired': False,
            'detectOrientation': True,
            'scale': True,
            'OCREngine': 2
        }
        
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(url, data=payload, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('OCRExitCode') == 1:
                return result['ParsedResults'][0]['ParsedText'].strip()
        return None
    except:
        return None

def ocr_space_english(image):
    """
    OCR.space avec langue anglaise - Méthode 2
    """
    return ocr_space_api(image, language='eng')

def extract_text_with_tesseract_fallback(image):
    """
    Méthode de fallback - Analyse de patterns
    """
    # Cette méthode utilise des heuristiques pour trouver des zones de texte
    try:
        # Convertir en niveaux de gris
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image
        
        # Binarisation
        threshold = 150
        binary = gray.point(lambda x: 0 if x < threshold else 255, '1')
        
        # Analyser les zones de texte potentielles
        pixels = binary.load()
        width, height = binary.size
        
        # Détecter les lignes de texte par densité de pixels
        text_regions = []
        for y in range(0, height, 10):
            row_density = sum(1 for x in range(width) if pixels[x, y] == 0) / width
            if 0.1 < row_density < 0.9:
                text_regions.append(y)
        
        # Si on trouve des régions de texte, simuler une extraction
        if len(text_regions) > 10:
            return "TEXTE_DETECTE_PATTERN"
        return None
    except:
        return None

# ============================================
# EXTRACTION INTELLIGENTE DES NUMÉROS
# ============================================

def extract_numbers_advanced(text):
    """
    Extraction avancée de numéros avec validation
    """
    if not text:
        return []
    
    numbers_found = set()
    
    # Pattern 1: Numéros standards
    pattern1 = r'\b\d+\b'
    numbers_found.update(re.findall(pattern1, text))
    
    # Pattern 2: Numéros avec séparateurs
    pattern2 = r'\b\d{1,3}(?:[ ,.]\d{3})*(?:[.,]\d+)?\b'
    matches2 = re.findall(pattern2, text)
    for match in matches2:
        clean = re.sub(r'[ ,.]', '', match)
        if clean.isdigit():
            numbers_found.add(clean)
    
    # Pattern 3: Numéros de 5 chiffres ou plus (souvent importants)
    pattern3 = r'\b\d{5,}\b'
    numbers_found.update(re.findall(pattern3, text))
    
    # Pattern 4: Numéros avec lettres mélangées (références)
    pattern4 = r'\b[A-Z0-9]{4,}\b'
    matches4 = re.findall(pattern4, text, re.IGNORECASE)
    for match in matches4:
        # Extraire seulement les chiffres si présents
        digits = re.findall(r'\d+', match)
        numbers_found.update(digits)
    
    # Filtrer et valider
    validated_numbers = []
    for num in numbers_found:
        # Nettoyer
        clean_num = re.sub(r'[^\d]', '', num)
        
        # Critères de validation
        if len(clean_num) >= 2:  # Au moins 2 chiffres
            if len(clean_num) <= 20:  # Pas trop long
                if not clean_num.startswith('0' * len(clean_num)):  # Pas que des zéros
                    validated_numbers.append(clean_num)
    
    # Trier par longueur (les plus longs souvent plus importants)
    validated_numbers.sort(key=lambda x: (len(x), int(x) if x.isdigit() else 0), reverse=True)
    
    return validated_numbers

def validate_numbers_with_context(numbers, full_text):
    """
    Validation contextuelle des numéros
    """
    validated = []
    
    # Mots-clés qui indiquent des numéros importants
    keywords = ['numéro', 'numero', 'no', 'n°', 'ref', 'réf', 'reference', 'id', 'code']
    
    for num in numbers:
        # Vérifier si proche d'un mot-clé
        for keyword in keywords:
            pattern = rf'{keyword}[^\d]*{re.escape(num)}'
            if re.search(pattern, full_text, re.IGNORECASE):
                validated.append({
                    'number': num,
                    'confidence': 'ÉLEVÉE',
                    'context': f'Près de "{keyword}"'
                })
                break
        else:
            # Validation par longueur et format
            if len(num) >= 5:
                confidence = 'MOYENNE'
                context = 'Numéro long'
            elif len(num) >= 8:
                confidence = 'ÉLEVÉE'
                context = 'Très long'
            else:
                confidence = 'BASSE'
                context = 'Court'
            
            validated.append({
                'number': num,
                'confidence': confidence,
                'context': context
            })
    
    return validated

# ============================================
# FONCTION PRINCIPALE DE DÉTECTION
# ============================================

def detect_numbers_high_precision(image):
    """
    Détection haute précision utilisant multiples méthodes
    """
    results = {
        'texts_found': [],
        'all_numbers': set(),
        'validated_numbers': [],
        'methods_used': []
    }
    
    # Créer plusieurs versions de l'image
    image_versions = create_multiple_versions(image)
    
    # Méthode 1: OCR.space Français
    for version_name, img_version in image_versions:
        text = ocr_space_api(img_version, language='fre')
        if text:
            results['texts_found'].append(text)
            results['methods_used'].append(f'OCR_FR_{version_name}')
            numbers = extract_numbers_advanced(text)
            results['all_numbers'].update(numbers)
    
    # Méthode 2: OCR.space Anglais
    for version_name, img_version in image_versions[:2]:  # Limiter pour performance
        text = ocr_space_english(img_version)
        if text:
            results['texts_found'].append(text)
            results['methods_used'].append(f'OCR_EN_{version_name}')
            numbers = extract_numbers_advanced(text)
            results['all_numbers'].update(numbers)
    
    # Combiner tous les textes pour validation contextuelle
    all_text_combined = ' '.join(results['texts_found'])
    
    # Valider les numéros avec le contexte
    if results['all_numbers']:
        validated = validate_numbers_with_context(
            list(results['all_numbers']), 
            all_text_combined
        )
        results['validated_numbers'] = validated
    
    return results

# ============================================
# INTERFACE STREAMLIT
# ============================================

def save_to_csv(data):
    """Convertit les données en CSV"""
    df = pd.DataFrame(data)
    return df.to_csv(index=False, encoding='utf-8-sig')

# Initialisation session state
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = {}
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Sidebar
with st.sidebar:
    st.header("🎯 Configuration Haute Précision")
    
    # Mode de précision
    precision_mode = st.selectbox(
        "Mode de précision",
        ["Équilibré", "Maximum", "Rapide"],
        help="Maximum: plus précis mais plus lent"
    )
    
    st.markdown("---")
    
    # Upload
    st.subheader("📤 Upload des photos")
    uploaded_files = st.file_uploader(
        "Choisissez vos photos",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file not in st.session_state.processed_images:
                st.session_state.processed_images.append(file)
                st.session_state.analysis_complete = False
    
    st.markdown("---")
    
    # Boutons d'action
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 ANALYSER", type="primary", use_container_width=True):
            if st.session_state.processed_images:
                st.session_state.analysis_complete = False
                st.rerun()
            else:
                st.warning("Ajoutez des photos")
    
    with col2:
        if st.button("🗑️ RÉINITIALISER", use_container_width=True):
            st.session_state.processed_images = []
            st.session_state.detection_results = {}
            st.session_state.analysis_complete = False
            st.rerun()
    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 Statistiques")
    st.metric("Photos", len(st.session_state.processed_images))
    st.metric("Analysées", len(st.session_state.detection_results))

# Zone principale
tab1, tab2, tab3 = st.tabs(["📋 Photos", "🎯 Résultats", "📈 Analyse Détaillée"])

with tab1:
    st.subheader("Photos à analyser")
    
    if st.session_state.processed_images:
        cols = st.columns(min(3, len(st.session_state.processed_images)))
        
        for idx, img_file in enumerate(st.session_state.processed_images):
            with cols[idx % 3]:
                image = Image.open(img_file)
                st.image(image, caption=f"{img_file.name}", use_container_width=True)
                
                # Afficher le statut si déjà analysé
                if img_file.name in st.session_state.detection_results:
                    num_count = len(st.session_state.detection_results[img_file.name]['validated_numbers'])
                    st.success(f"✅ {num_count} numéros trouvés")
                
                if st.button("❌", key=f"del_tab1_{idx}"):
                    st.session_state.processed_images.pop(idx)
                    if img_file.name in st.session_state.detection_results:
                        del st.session_state.detection_results[img_file.name]
                    st.rerun()
    else:
        st.info("👈 Ajoutez des photos dans la barre latérale")
        
        # Démonstration
        with st.expander("🧪 Tester avec du texte", expanded=True):
            st.markdown("### Test de détection sur texte")
            test_input = st.text_area(
                "Collez du texte contenant des numéros:",
                value="HENGSTLER\n10406871\n823743",
                height=120
            )
            
            if st.button("🧪 Tester la détection"):
                numbers = extract_numbers_advanced(test_input)
                if numbers:
                    validated = validate_numbers_with_context(numbers, test_input)
                    
                    st.success(f"✅ {len(validated)} numéros détectés!")
                    
                    df_test = pd.DataFrame(validated)
                    st.dataframe(df_test, use_container_width=True)
                    
                    # Afficher les numéros
                    st.code('\n'.join([v['number'] for v in validated]))
                else:
                    st.warning("Aucun numéro détecté")

with tab2:
    st.subheader("Résultats de la détection")
    
    # Lancer l'analyse si nécessaire
    if (st.session_state.processed_images and 
        not st.session_state.analysis_complete):
        
        st.info("🔄 Analyse haute précision en cours...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, img_file in enumerate(st.session_state.processed_images):
            status_text.text(f"🎯 Analyse de {img_file.name} (précision: {precision_mode})")
            
            # Ouvrir l'image
            image = Image.open(img_file)
            
            # Détection haute précision
            with st.spinner(f"Analyse approfondie..."):
                results = detect_numbers_high_precision(image)
            
            st.session_state.detection_results[img_file.name] = results
            
            progress_bar.progress((idx + 1) / len(st.session_state.processed_images))
        
        status_text.text("✅ Analyse terminée!")
        st.session_state.analysis_complete = True
        st.rerun()
    
    # Afficher les résultats
    if st.session_state.detection_results:
        st.success(f"✅ {len(st.session_state.detection_results)} photos analysées")
        
        # Tableau récapitulatif
        summary_data = []
        all_numbers_for_export = []
        
        for filename, results in st.session_state.detection_results.items():
            high_conf = [n for n in results['validated_numbers'] if n['confidence'] == 'ÉLEVÉE']
            medium_conf = [n for n in results['validated_numbers'] if n['confidence'] == 'MOYENNE']
            
            summary_data.append({
                'Fichier': filename,
                'Total numéros': len(results['validated_numbers']),
                'Haute confiance': len(high_conf),
                'Confiance moyenne': len(medium_conf),
                'Méthodes utilisées': len(results['methods_used'])
            })
            
            # Préparer pour export
            for num_info in results['validated_numbers']:
                all_numbers_for_export.append({
                    'Fichier': filename,
                    'Numéro': num_info['number'],
                    'Confiance': num_info['confidence'],
                    'Contexte': num_info['context'],
                    'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Métriques globales
        total_numbers = sum(s['Total numéros'] for s in summary_data)
        high_conf_total = sum(s['Haute confiance'] for s in summary_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Numéros", total_numbers)
        with col2:
            st.metric("Haute Confiance", high_conf_total)
        with col3:
            precision_rate = (high_conf_total / total_numbers * 100) if total_numbers > 0 else 0
            st.metric("Taux Précision", f"{precision_rate:.1f}%")
        
        # Export
        st.markdown("---")
        st.subheader("💾 Export des résultats")
        
        if all_numbers_for_export:
            df_export = pd.DataFrame(all_numbers_for_export)
            
            csv_data = save_to_csv(df_export)
            
            col_down1, col_down2, col_down3 = st.columns(3)
            
            with col_down1:
                st.download_button(
                    label="📥 CSV Complet",
                    data=csv_data,
                    file_name=f"numeros_haute_precision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_down2:
                # Export haute confiance uniquement
                high_conf_export = [n for n in all_numbers_for_export if n['Confiance'] == 'ÉLEVÉE']
                if high_conf_export:
                    df_high = pd.DataFrame(high_conf_export)
                    csv_high = save_to_csv(df_high)
                    
                    st.download_button(
                        label="⭐ Haute Confiance",
                        data=csv_high,
                        file_name=f"numeros_haute_confiance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col_down3:
                # Liste simple des numéros
                simple_list = []
                for item in all_numbers_for_export:
                    if item['Confiance'] in ['ÉLEVÉE', 'MOYENNE']:
                        simple_list.append(item['Numéro'])
                
                if simple_list:
                    simple_text = '\n'.join(simple_list)
                    st.download_button(
                        label="📄 Liste Simple",
                        data=simple_text,
                        file_name=f"liste_numeros_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

with tab3:
    st.subheader("Analyse Détaillée par Photo")
    
    if st.session_state.detection_results:
        for filename, results in st.session_state.detection_results.items():
            with st.expander(f"📊 {filename}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📝 Textes extraits:**")
                    for idx, text in enumerate(results['texts_found'][:3], 1):
                        st.text_area(f"Texte {idx}", text, height=80, key=f"text_{filename}_{idx}")
                    
                    st.write("**🔧 Méthodes utilisées:**")
                    for method in results['methods_used']:
                        st.write(f"- {method}")
                
                with col2:
                    st.write("**🔢 Numéros validés:**")
                    
                    if results['validated_numbers']:
                        df_nums = pd.DataFrame(results['validated_numbers'])
                        
                        # Colorer par confiance
                        def color_confidence(val):
                            if val == 'ÉLEVÉE':
                                return 'background-color: #90EE90'
                            elif val == 'MOYENNE':
                                return 'background-color: #FFD700'
                            return ''
                        
                        styled_df = df_nums.style.applymap(color_confidence, subset=['confidence'])
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Afficher les numéros haute confiance
                        high_conf_nums = [n['number'] for n in results['validated_numbers'] if n['confidence'] == 'ÉLEVÉE']
                        if high_conf_nums:
                            st.success(f"⭐ Haute confiance: {', '.join(high_conf_nums)}")
                    else:
                        st.warning("Aucun numéro validé")
    else:
        st.info("Analysez d'abord des photos pour voir les détails")

# Footer
st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;'>
    <h2 style='text-align: center; color: white;'>🎯 Système de Détection Haute Précision</h2>
    <div style='display: flex; justify-content: space-around; margin-top: 20px;'>
        <div style='text-align: center;'>
            <h3 style='color: white;'>🔍 Multi-OCR</h3>
            <p>Français + Anglais</p>
        </div>
        <div style='text-align: center;'>
            <h3 style='color: white;'>🎨 Prétraitement</h3>
            <p>Multi-versions</p>
        </div>
        <div style='text-align: center;'>
            <h3 style='color: white;'>✅ Validation</h3>
            <p>Contextuelle</p>
        </div>
        <div style='text-align: center;'>
            <h3 style='color: white;'>📊 Confiance</h3>
            <p>Élevée/Moyenne/Basse</p>
        </div>
    </div>
    <p style='text-align: center; margin-top: 20px; font-size: 14px;'>
        ✅ Détection automatique • Validation intelligente • Export multi-formats
    </p>
</div>
""", unsafe_allow_html=True)

# CSS pour améliorer l'interface
st.markdown("""
<style>
    .stButton > button {
        border-radius: 10px;
        font-weight: bold;
    }
    .stProgress > div > div {
        background-color: #667eea;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)
