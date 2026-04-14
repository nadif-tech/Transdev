import streamlit as st
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
from datetime import datetime
import pytesseract
import os

# Configuration de la page
st.set_page_config(
    page_title="Extracteur Numeros Compteur",
    page_icon="123",
    layout="wide"
)

st.title("Extracteur de Numeros - Compteur Horaire/Kilometrage")
st.markdown("---")

# Configuration Tesseract pour Streamlit Cloud
if os.path.exists('/usr/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
elif os.path.exists('/usr/local/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

def preprocess_counter_image(image):
    """Pretraitement optimise pour les photos de compteurs"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    if width < 1500:
        ratio = 1500 / width
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    gray = image.convert('L')
    
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(3.5)
    
    enhancer = ImageEnhance.Sharpness(gray)
    gray = enhancer.enhance(3.0)
    
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    
    return gray, image

def create_binary_versions(gray_image):
    """Cree differentes versions binaires"""
    versions = []
    
    v1 = gray_image.point(lambda x: 0 if x < 150 else 255, '1')
    versions.append(v1)
    
    v2 = gray_image.point(lambda x: 0 if x < 180 else 255, '1')
    versions.append(v2)
    
    v3 = gray_image.point(lambda x: 0 if x < 120 else 255, '1')
    versions.append(v3)
    
    v4 = gray_image.point(lambda x: 255 - x)
    v4 = v4.point(lambda x: 0 if x < 150 else 255, '1')
    versions.append(v4)
    
    return versions

def extract_numbers_from_image(image):
    """Extrait tous les numeros avec Tesseract"""
    all_numbers = set()
    all_text = []
    
    gray, original = preprocess_counter_image(image)
    binary_versions = create_binary_versions(gray)
    
    configs = [
        '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789',
        '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789',
        '--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789',
        '--psm 6 --oem 3',
    ]
    
    for version in binary_versions:
        for config in configs:
            try:
                text = pytesseract.image_to_string(version, config=config, lang='fra+eng')
                if text:
                    all_text.append(text)
                    numbers = re.findall(r'\b\d{4,}\b', text)
                    all_numbers.update(numbers)
                    numbers2 = re.findall(r'\b\d[\d\s]{3,}\d\b', text)
                    for num in numbers2:
                        clean = re.sub(r'\s+', '', num)
                        if clean.isdigit() and len(clean) >= 4:
                            all_numbers.add(clean)
            except:
                pass
    
    try:
        text_gray = pytesseract.image_to_string(gray, config='--psm 6 --oem 3', lang='fra+eng')
        if text_gray:
            all_text.append(text_gray)
            numbers = re.findall(r'\b\d{4,}\b', text_gray)
            all_numbers.update(numbers)
    except:
        pass
    
    return list(all_numbers), ' '.join(all_text)

def classify_numbers(numbers, full_text):
    """Classe les numeros en heures et kilometrages"""
    heure_numbers = []
    km_numbers = []
    autres = []
    
    text_lower = full_text.lower()
    
    for num in numbers:
        if 'heure' in text_lower or 'hour' in text_lower or 'time' in text_lower:
            heure_numbers.append(num)
        elif 'km' in text_lower or 'kilomet' in text_lower or 'mile' in text_lower:
            km_numbers.append(num)
        else:
            if 7 <= len(num) <= 9:
                heure_numbers.append(num)
            elif 4 <= len(num) <= 6:
                km_numbers.append(num)
            else:
                autres.append(num)
    
    return {
        'heures': heure_numbers,
        'kilometrages': km_numbers,
        'autres': autres
    }

# Session state
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'results' not in st.session_state:
    st.session_state.results = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Sidebar
with st.sidebar:
    st.header("Upload Photos Compteur")
    
    uploaded_files = st.file_uploader(
        "Choisissez vos photos",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.session_state.uploaded_images = uploaded_files
        st.success(f"{len(uploaded_files)} photo(s) chargee(s)")
    
    st.markdown("---")
    
    st.header("Parametres")
    
    min_digits = st.slider(
        "Nombre minimum de chiffres",
        min_value=3,
        max_value=10,
        value=4
    )
    
    st.markdown("---")
    
    st.header("Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("ANALYSER", type="primary", use_container_width=True):
            if not st.session_state.uploaded_images:
                st.warning("Chargez des photos")
            else:
                st.session_state.processing = True
                st.session_state.results = []
                st.rerun()
    
    with col2:
        if st.button("EFFACER", use_container_width=True):
            st.session_state.uploaded_images = []
            st.session_state.results = []
            st.rerun()
    
    st.markdown("---")
    
    st.header("Statistiques")
    st.metric("Photos", len(st.session_state.uploaded_images))
    st.metric("Analyses", len(st.session_state.results))

# Zone principale
tab1, tab2, tab3 = st.tabs(["Photos", "Analyse", "Resultats"])

with tab1:
    st.header("Photos de compteurs")
    
    if st.session_state.uploaded_images:
        cols = st.columns(min(2, len(st.session_state.uploaded_images)))
        
        for idx, img_file in enumerate(st.session_state.uploaded_images):
            with cols[idx % 2]:
                image = Image.open(img_file)
                st.image(image, caption=img_file.name, use_container_width=True)
                
                if st.button(f"Supprimer", key=f"del_{idx}"):
                    st.session_state.uploaded_images.pop(idx)
                    st.rerun()
    else:
        st.info("Chargez des photos de compteurs dans la barre laterale")
        st.markdown("---")
        st.markdown("### Exemple de photo attendue")
        st.write("Contenu typique:")
        st.code("Nombre d'heure\n10406871\nKilometrage\n623743")
        st.write("Numeros qui seront detectes: 10406871 (Heures) et 623743 (Kilometrage)")

with tab2:
    st.header("Analyse OCR en cours")
    
    if st.session_state.processing:
        st.info("Traitement des photos...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, img_file in enumerate(st.session_state.uploaded_images):
            status_text.text(f"Analyse de {img_file.name} ({idx+1}/{len(st.session_state.uploaded_images)})")
            
            image = Image.open(img_file)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption=f"Original: {img_file.name}", use_container_width=True)
            
            with col2:
                with st.spinner("Tesseract OCR en cours..."):
                    numbers, full_text = extract_numbers_from_image(image)
                    numbers = [n for n in numbers if len(n) >= min_digits]
                    classified = classify_numbers(numbers, full_text)
                    
                    result = {
                        'filename': img_file.name,
                        'heures': classified['heures'],
                        'kilometrages': classified['kilometrages'],
                        'autres': classified['autres'],
                        'all_numbers': numbers,
                        'full_text': full_text,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    if classified['heures']:
                        st.success(f"Heures: {', '.join(classified['heures'])}")
                    if classified['kilometrages']:
                        st.success(f"Kilometrage: {', '.join(classified['kilometrages'])}")
                    
                    if not numbers:
                        st.warning("Aucun numero detecte")
                    
                    st.session_state.results.append(result)
            
            progress_bar.progress((idx + 1) / len(st.session_state.uploaded_images))
        
        status_text.text("Analyse terminee!")
        st.session_state.processing = False
        st.success("Toutes les photos ont ete analysees!")
        st.rerun()
    
    else:
        if st.session_state.uploaded_images:
            st.info("Cliquez sur 'ANALYSER' dans la barre laterale")
        else:
            st.info("Uploadez des photos")

with tab3:
    st.header("Resultats de l'extraction")
    
    if st.session_state.results:
        st.subheader("Resume des extractions")
        
        summary_data = []
        all_heures = []
        all_km = []
        
        for r in st.session_state.results:
            summary_data.append({
                'Fichier': r['filename'],
                'Heures': len(r['heures']),
                'Kilometrages': len(r['kilometrages']),
                'Autres': len(r['autres']),
                'Date': r['timestamp']
            })
            all_heures.extend(r['heures'])
            all_km.extend(r['kilometrages'])
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Fichiers", len(st.session_state.results))
        with col2:
            st.metric("Heures", len(all_heures))
        with col3:
            st.metric("Kilometrages", len(all_km))
        with col4:
            st.metric("Total numeros", len(all_heures) + len(all_km))
        
        st.markdown("---")
        
        st.subheader("Details par photo")
        
        for idx, result in enumerate(st.session_state.results):
            with st.expander(f"{result['filename']}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    img_file = st.session_state.uploaded_images[idx] if idx < len(st.session_state.uploaded_images) else None
                    if img_file:
                        image = Image.open(img_file)
                        st.image(image, width=250)
                
                with col2:
                    st.write(f"Date: {result['timestamp']}")
                    
                    if result['heures']:
                        st.success("Nombres d'heures:")
                        for num in result['heures']:
                            st.code(num, language='text')
                    
                    if result['kilometrages']:
                        st.success("Kilometrages:")
                        for num in result['kilometrages']:
                            st.code(num, language='text')
        
        st.markdown("---")
        st.subheader("Export des resultats")
        
        export_data = []
        for r in st.session_state.results:
            for num in r['heures']:
                export_data.append({
                    'Fichier': r['filename'],
                    'Type': 'Heures',
                    'Valeur': num,
                    'Date': r['timestamp']
                })
            for num in r['kilometrages']:
                export_data.append({
                    'Fichier': r['filename'],
                    'Type': 'Kilometrage',
                    'Valeur': num,
                    'Date': r['timestamp']
                })
        
        if export_data:
            df_export = pd.DataFrame(export_data)
            csv_data = df_export.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="TELECHARGER CSV",
                data=csv_data,
                file_name=f"compteur_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.info("Aucun resultat disponible. Lancez une analyse.")

# Footer
st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;'>
    <h3 style='color: white;'>Extracteur de Numeros - Compteurs</h3>
    <p>Detection automatique des heures et kilometrages</p>
    <p style='font-size: 12px;'>Tesseract OCR | Compatible Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)
