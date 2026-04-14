import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import io
import re
from datetime import datetime
import cv2
import easyocr

# Configuration de la page
st.set_page_config(
    page_title="Extracteur de Numeros - OCR",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 Extracteur de Numeros depuis Photos")
st.markdown("---")

# Initialisation du reader EasyOCR
@st.cache_resource
def load_ocr_reader():
    """Charge le reader EasyOCR une seule fois"""
    return easyocr.Reader(['fr', 'en'], gpu=False)

def preprocess_image(image):
    """Pretraitement de l'image pour ameliorer l'OCR"""
    # Convertir PIL en OpenCV
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convertir en niveaux de gris
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Ameliorer le contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Debruitage
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # Binarisation adaptative
    binary = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return binary

def extract_numbers_from_text(text):
    """Extrait tous les nombres du texte"""
    if not text:
        return []
    
    # Patterns pour differents formats
    patterns = [
        r'\b\d{5,}\b',  # 5 chiffres ou plus
        r'\b\d{2,4}\b',  # 2-4 chiffres
        r'\b\d+[-\s]?\d+\b',  # Avec separateurs
    ]
    
    all_numbers = set()
    for pattern in patterns:
        numbers = re.findall(pattern, text)
        for num in numbers:
            clean_num = re.sub(r'[-\s]', '', num)
            if clean_num.isdigit() and len(clean_num) >= 2:
                all_numbers.add(clean_num)
    
    return sorted(list(all_numbers), key=len, reverse=True)

def process_image_with_easyocr(image, reader):
    """Traite une image avec EasyOCR"""
    try:
        # Pretraitement
        processed = preprocess_image(image)
        
        # OCR
        results = reader.readtext(processed, detail=1, paragraph=False)
        
        # Extraire le texte
        full_text = ' '.join([item[1] for item in results])
        
        # Extraire les numeros
        numbers = extract_numbers_from_text(full_text)
        
        # Preparer les details avec confiance
        details = []
        for bbox, text, confidence in results:
            if confidence > 0.3:  # Filtrer par confiance
                details.append({
                    'text': text,
                    'confidence': round(confidence, 2),
                    'bbox': bbox
                })
        
        return {
            'success': True,
            'full_text': full_text,
            'numbers': numbers,
            'details': details,
            'count': len(numbers)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'numbers': [],
            'full_text': '',
            'details': [],
            'count': 0
        }

def draw_boxes_on_image(image, details):
    """Dessine les bounding boxes sur l'image"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    img_copy = image.copy()
    
    for item in details:
        bbox = item['bbox']
        text = item['text']
        conf = item['confidence']
        
        # Points du rectangle
        pts = np.array(bbox, dtype=np.int32)
        
        # Couleur selon confiance
        if conf > 0.7:
            color = (0, 255, 0)  # Vert
        elif conf > 0.5:
            color = (255, 255, 0)  # Jaune
        else:
            color = (255, 0, 0)  # Rouge
        
        cv2.polylines(img_copy, [pts], True, color, 2)
        
        # Ajouter le texte
        x, y = int(bbox[0][0]), int(bbox[0][1]) - 10
        cv2.putText(img_copy, f"{text} ({conf:.2f})", 
                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 1)
    
    return img_copy

def create_excel_export(results):
    """Cree un fichier Excel avec les resultats"""
    data = []
    for result in results:
        for num in result.get('numbers', []):
            data.append({
                'Fichier': result['filename'],
                'Numero': num,
                'Date': result['timestamp'],
                'Texte complet': result['full_text'][:200]
            })
    
    if data:
        df = pd.DataFrame(data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Numeros', index=False)
        output.seek(0)
        return output
    return None

# Session state
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'results' not in st.session_state:
    st.session_state.results = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Sidebar
with st.sidebar:
    st.header("📤 Upload des photos")
    
    uploaded_files = st.file_uploader(
        "Choisissez vos photos",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
        accept_multiple_files=True,
        help="Formats supportes: PNG, JPG, JPEG, BMP, TIFF, WEBP"
    )
    
    if uploaded_files:
        st.session_state.uploaded_images = uploaded_files
        st.success(f"✅ {len(uploaded_files)} photo(s) chargee(s)")
    
    st.markdown("---")
    
    st.header("⚙️ Options")
    
    confidence_threshold = st.slider(
        "Seuil de confiance",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.1,
        help="Filtre les resultats selon la confiance"
    )
    
    show_boxes = st.checkbox("Afficher les zones detectees", value=True)
    
    st.markdown("---")
    
    st.header("🎯 Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 ANALYSER", type="primary", use_container_width=True):
            if not st.session_state.uploaded_images:
                st.warning("⚠️ Chargez d'abord des photos")
            else:
                st.session_state.processing = True
                st.session_state.results = []
                st.rerun()
    
    with col2:
        if st.button("🗑️ EFFACER", use_container_width=True):
            st.session_state.uploaded_images = []
            st.session_state.results = []
            st.session_state.processing = False
            st.rerun()
    
    st.markdown("---")
    
    st.header("📊 Statistiques")
    st.metric("Photos", len(st.session_state.uploaded_images))
    st.metric("Analyses", len(st.session_state.results))
    
    if st.session_state.results:
        total = sum(r['count'] for r in st.session_state.results)
        st.metric("Numeros trouves", total)

# Zone principale
tab1, tab2, tab3 = st.tabs(["📋 Galerie", "🔬 Analyse", "📈 Resultats"])

with tab1:
    st.header("Galerie des photos")
    
    if st.session_state.uploaded_images:
        cols = st.columns(min(3, len(st.session_state.uploaded_images)))
        
        for idx, img_file in enumerate(st.session_state.uploaded_images):
            with cols[idx % 3]:
                image = Image.open(img_file)
                st.image(image, caption=img_file.name, use_container_width=True)
                
                st.caption(f"📏 {image.size[0]}x{image.size[1]}")
                
                if st.button(f"❌ Supprimer", key=f"del_{idx}"):
                    st.session_state.uploaded_images.pop(idx)
                    st.rerun()
    else:
        st.info("👈 Chargez des photos via la barre laterale")
        
        st.markdown("---")
        st.markdown("### 📌 Comment utiliser")
        st.markdown("""
        1. **Uploadez** vos photos dans la barre laterale
        2. **Cliquez** sur "ANALYSER"
        3. **Visualisez** les numeros extraits
        4. **Telechargez** les resultats en Excel
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Exemple (TESTO01.jpeg)")
        st.code("HENGSTLER\n10406871\n823743")
        st.success("Ces numeros seront automatiquement detectes!")

with tab2:
    st.header("Analyse OCR en cours")
    
    if st.session_state.processing:
        st.info("🔄 Traitement des photos...")
        
        # Charger le reader OCR
        with st.spinner("Chargement du modele EasyOCR..."):
            reader = load_ocr_reader()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, img_file in enumerate(st.session_state.uploaded_images):
            status_text.text(f"📸 Analyse de {img_file.name} ({idx+1}/{len(st.session_state.uploaded_images)})")
            
            image = Image.open(img_file)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption=f"Original: {img_file.name}", use_container_width=True)
            
            with col2:
                with st.spinner("🔍 OCR en cours..."):
                    result = process_image_with_easyocr(image, reader)
                    
                    result['filename'] = img_file.name
                    result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    result['image_size'] = f"{image.size[0]}x{image.size[1]}"
                    
                    if result['success']:
                        st.success(f"✅ {result['count']} numeros trouves")
                        
                        if result['numbers']:
                            st.write("**🔢 Numeros detectes:**")
                            for num in result['numbers'][:10]:
                                st.code(num)
                        
                        if show_boxes and result['details']:
                            img_with_boxes = draw_boxes_on_image(image, result['details'])
                            st.image(img_with_boxes, caption="Zones detectees", use_container_width=True)
                        
                        st.session_state.results.append(result)
                    else:
                        st.error(f"❌ Erreur: {result['error']}")
            
            progress_bar.progress((idx + 1) / len(st.session_state.uploaded_images))
        
        status_text.text("✅ Analyse terminee!")
        st.session_state.processing = False
        
        st.success("🎉 Toutes les photos ont ete analysees!")
        st.rerun()
    
    else:
        if st.session_state.uploaded_images:
            st.info("👆 Cliquez sur 'ANALYSER' dans la barre laterale pour commencer")
            
            st.markdown("### 📸 Apercu des photos a analyser")
            cols = st.columns(min(4, len(st.session_state.uploaded_images)))
            
            for idx, img_file in enumerate(st.session_state.uploaded_images):
                with cols[idx % 4]:
                    image = Image.open(img_file)
                    st.image(image, caption=img_file.name, width=150)
        else:
            st.info("📤 Uploadez d'abord des photos dans la barre laterale")

with tab3:
    st.header("Resultats de l'extraction")
    
    if st.session_state.results:
        # Resume
        st.subheader("📊 Resume")
        
        summary_data = []
        all_numbers = []
        
        for r in st.session_state.results:
            summary_data.append({
                'Fichier': r['filename'],
                'Numeros': r['count'],
                'Taille': r['image_size'],
                'Date': r['timestamp']
            })
            all_numbers.extend(r.get('numbers', []))
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Metriques
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📁 Fichiers", len(st.session_state.results))
        with col2:
            st.metric("🔢 Total numeros", len(all_numbers))
        with col3:
            unique = len(set(all_numbers))
            st.metric("🎯 Numeros uniques", unique)
        with col4:
            avg = round(len(all_numbers) / len(st.session_state.results), 1) if st.session_state.results else 0
            st.metric("📊 Moyenne/fichier", avg)
        
        st.markdown("---")
        
        # Details par fichier
        st.subheader("📋 Details par photo")
        
        for idx, result in enumerate(st.session_state.results):
            with st.expander(f"📄 {result['filename']} - {result['count']} numeros"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    img_file = st.session_state.uploaded_images[idx] if idx < len(st.session_state.uploaded_images) else None
                    if img_file:
                        image = Image.open(img_file)
                        st.image(image, width=200)
                        
                        if show_boxes and result.get('details'):
                            img_boxes = draw_boxes_on_image(image, result['details'])
                            st.image(img_boxes, caption="Avec detections", width=200)
                
                with col2:
                    st.write(f"**Date:** {result['timestamp']}")
                    st.write(f"**Taille:** {result['image_size']}")
                    
                    if result.get('numbers'):
                        st.write("**🔢 Numeros detectes:**")
                        for num in result['numbers']:
                            st.code(num)
                    else:
                        st.warning("Aucun numero detecte")
                    
                    if result.get('full_text'):
                        st.text_area(
                            "📝 Texte complet",
                            result['full_text'][:300],
                            height=80,
                            key=f"text_{idx}"
                        )
                    
                    if result.get('details'):
                        st.write(f"**📍 Zones detectees:** {len(result['details'])}")
        
        st.markdown("---")
        
        # Export
        st.subheader("💾 Export des resultats")
        
        export_format = st.radio(
            "Format",
            ["Excel", "CSV"],
            horizontal=True
        )
        
        if export_format == "Excel":
            excel_file = create_excel_export(st.session_state.results)
            if excel_file:
                st.download_button(
                    label="📥 TELECHARGER EXCEL",
                    data=excel_file,
                    file_name=f"numeros_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            data = []
            for r in st.session_state.results:
                for num in r.get('numbers', []):
                    data.append({
                        'Fichier': r['filename'],
                        'Numero': num,
                        'Date': r['timestamp']
                    })
            
            if data:
                df = pd.DataFrame(data)
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                
                st.download_button(
                    label="📥 TELECHARGER CSV",
                    data=csv,
                    file_name=f"numeros_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Copier tous les numeros
        if all_numbers:
            st.markdown("---")
            st.subheader("📋 Copier tous les numeros")
            
            numbers_text = '\n'.join(all_numbers)
            st.text_area(
                "Tous les numeros (Ctrl+C pour copier)",
                numbers_text,
                height=150
            )
    
    else:
        st.info("🔍 Aucun resultat disponible. Lancez une analyse dans l'onglet 'Analyse'.")

# Footer
st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;'>
    <h3 style='color: white;'>🔢 Extracteur de Numeros avec EasyOCR</h3>
    <p>Detection automatique • Multi-langues (FR/EN) • Export Excel/CSV</p>
    <p style='font-size: 12px; margin-top: 10px;'>Compatible Streamlit Cloud | EasyOCR + OpenCV</p>
</div>
""", unsafe_allow_html=True)
