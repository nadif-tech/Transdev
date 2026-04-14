import streamlit as st
import pandas as pd
from PIL import Image
import io
import re
from datetime import datetime
import requests
import json
import time

# Configuration de la page
st.set_page_config(
    page_title="Extracteur de Numeros",
    page_icon="123",
    layout="wide"
)

st.title("Extracteur de Numeros - OCR Gratuit")
st.markdown("---")

def extract_text_with_ocr_space(image):
    """Extrait le texte avec OCR.space (API gratuite)"""
    try:
        img_byte_arr = io.BytesIO()
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(img_byte_arr, format='JPEG', quality=95)
        img_bytes = img_byte_arr.getvalue()
        
        url = "https://api.ocr.space/parse/image"
        
        payload = {
            'apikey': 'K86742198888957',
            'language': 'fre',
            'isOverlayRequired': False,
            'detectOrientation': True,
            'scale': True,
            'OCREngine': 2
        }
        
        files = {
            'file': ('image.jpg', img_bytes, 'image/jpeg')
        }
        
        response = requests.post(url, data=payload, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('OCRExitCode') == 1:
                text = result['ParsedResults'][0]['ParsedText']
                return text.strip()
        return None
    except Exception as e:
        st.error(f"Erreur OCR: {str(e)}")
        return None

def extract_numbers(text):
    """Extrait tous les nombres du texte"""
    if not text:
        return []
    
    numbers = re.findall(r'\b\d+\b', text)
    
    valid_numbers = []
    for num in numbers:
        if len(num) >= 2 and len(num) <= 20:
            valid_numbers.append(num)
    
    return list(set(valid_numbers))

def create_csv_export(results):
    """Cree un fichier CSV avec les resultats"""
    data = []
    for result in results:
        for number in result.get('numbers', []):
            data.append({
                'Fichier': result['filename'],
                'Numero': number,
                'Date': result['timestamp']
            })
    
    if data:
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8-sig')
    return ""

# Session state
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'results' not in st.session_state:
    st.session_state.results = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Sidebar
with st.sidebar:
    st.header("Upload d'images")
    
    uploaded_files = st.file_uploader(
        "Choisissez vos photos",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
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
                st.warning("Chargez d'abord des images")
            else:
                st.session_state.processing = True
                st.session_state.results = []
                st.rerun()
    
    with col2:
        if st.button("EFFACER", use_container_width=True):
            st.session_state.uploaded_images = []
            st.session_state.results = []
            st.session_state.processing = False
            st.rerun()
    
    st.markdown("---")
    
    st.header("Statistiques")
    st.metric("Images", len(st.session_state.uploaded_images))
    st.metric("Resultats", len(st.session_state.results))

# Zone principale
tab1, tab2, tab3 = st.tabs(["Galerie", "Analyse", "Resultats"])

# Tab 1: Galerie
with tab1:
    st.header("Galerie d'images")
    
    if st.session_state.uploaded_images:
        cols = st.columns(min(3, len(st.session_state.uploaded_images)))
        
        for idx, img_file in enumerate(st.session_state.uploaded_images):
            with cols[idx % 3]:
                image = Image.open(img_file)
                st.image(image, caption=img_file.name, use_container_width=True)
                
                if st.button(f"Supprimer", key=f"del_{idx}"):
                    st.session_state.uploaded_images.pop(idx)
                    st.rerun()
    else:
        st.info("Chargez des images dans la barre laterale")
        
        st.markdown("---")
        st.markdown("### Exemple")
        st.write("Pour TESTO01.jpeg, l'application detectera:")
        st.code("10406871\n823743")

# Tab 2: Analyse
with tab2:
    st.header("Analyse OCR")
    
    if st.session_state.processing:
        st.info("Traitement en cours...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, img_file in enumerate(st.session_state.uploaded_images):
            status_text.text(f"Analyse de {img_file.name}")
            
            image = Image.open(img_file)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption=img_file.name, width=200)
            
            with col2:
                with st.spinner("OCR en cours..."):
                    text = extract_text_with_ocr_space(image)
                    
                    if text:
                        numbers = extract_numbers(text)
                        
                        result = {
                            'filename': img_file.name,
                            'numbers': numbers,
                            'full_text': text,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        st.success(f"{len(numbers)} numeros trouves")
                        
                        if numbers:
                            st.write("Numeros detectes:")
                            for num in numbers:
                                st.code(num)
                        
                        st.session_state.results.append(result)
                    else:
                        st.error("Echec de l'analyse")
                        
                        result = {
                            'filename': img_file.name,
                            'numbers': [],
                            'full_text': "",
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        st.session_state.results.append(result)
            
            progress_bar.progress((idx + 1) / len(st.session_state.uploaded_images))
            time.sleep(1)
        
        status_text.text("Analyse terminee!")
        st.session_state.processing = False
        st.rerun()
    
    else:
        if st.session_state.uploaded_images:
            st.info("Cliquez sur ANALYSER pour commencer")
            
            cols = st.columns(min(3, len(st.session_state.uploaded_images)))
            for idx, img_file in enumerate(st.session_state.uploaded_images):
                with cols[idx % 3]:
                    image = Image.open(img_file)
                    st.image(image, caption=img_file.name, width=150)
        else:
            st.info("Chargez d'abord des images")

# Tab 3: Resultats
with tab3:
    st.header("Resultats")
    
    if st.session_state.results:
        summary_data = []
        all_numbers = []
        
        for r in st.session_state.results:
            numbers_count = len(r.get('numbers', []))
            summary_data.append({
                'Fichier': r['filename'],
                'Numeros': numbers_count,
                'Date': r['timestamp']
            })
            all_numbers.extend(r.get('numbers', []))
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Fichiers", len(st.session_state.results))
        with col2:
            st.metric("Total Numeros", len(all_numbers))
        
        st.markdown("---")
        
        for idx, result in enumerate(st.session_state.results):
            with st.expander(f"{result['filename']} - {len(result.get('numbers', []))} numeros"):
                if result.get('numbers'):
                    st.write("Numeros detectes:")
                    for num in result['numbers']:
                        st.code(num)
                else:
                    st.warning("Aucun numero detecte")
                
                if result.get('full_text'):
                    st.text_area("Texte complet", result['full_text'][:300], height=80)
        
        st.markdown("---")
        
        if all_numbers:
            csv_data = create_csv_export(st.session_state.results)
            
            st.download_button(
                label="TELECHARGER CSV",
                data=csv_data,
                file_name=f"numeros_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.text_area("Tous les numeros", '\n'.join(all_numbers), height=100)
    
    else:
        st.info("Aucun resultat. Lancez une analyse.")

# Footer
st.markdown("---")
st.markdown("""
<div style='background: #1e3c72; padding: 20px; border-radius: 10px; color: white; text-align: center;'>
    <h3 style='color: white;'>Extracteur de Numeros OCR</h3>
    <p>API OCR.space gratuite | 25000 requetes/mois | Compatible Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)
