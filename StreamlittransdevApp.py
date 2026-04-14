import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
import numpy as np
import io
import re
from datetime import datetime
import os

# Configuration de la page
st.set_page_config(
    page_title="Extracteur de Numéros - OCR",
    page_icon="📸",
    layout="wide"
)

# Titre de l'application
st.title("📸 Extracteur de Numéros depuis Photos")
st.markdown("---")

# Fonction pour extraire les numéros du texte
def extract_numbers(text):
    """Extrait tous les numéros trouvés dans le texte"""
    if not text:
        return []
    # Pattern pour trouver des numéros (entiers ou décimaux)
    pattern = r'\b\d+(?:[.,]\d+)?\b'
    numbers = re.findall(pattern, text)
    return numbers

# Fonction pour traiter une image
def process_image(image):
    """Traite une image et extrait les numéros avec Tesseract"""
    try:
        # Convertir en niveaux de gris pour meilleure reconnaissance
        if image.mode != 'L':
            image = image.convert('L')
        
        # Configuration Tesseract pour le français
        custom_config = r'--oem 3 --psm 6 -l fra+eng'
        
        # OCR sur l'image
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Nettoyer le texte
        text = text.strip()
        
        # Extraire les numéros
        numbers = extract_numbers(text)
        
        return {
            'text': text,
            'numbers': numbers,
            'all_numbers': ', '.join(numbers) if numbers else '',
            'count': len(numbers),
            'success': True
        }
    except Exception as e:
        return {
            'text': '',
            'numbers': [],
            'all_numbers': '',
            'count': 0,
            'success': False,
            'error': str(e)
        }

# Fonction pour sauvegarder dans Excel
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

# Sidebar pour les options
with st.sidebar:
    st.header("⚙️ Options")
    
    # Upload multiple de photos
    uploaded_files = st.file_uploader(
        "Choisissez vos photos",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=True
    )
    
    # Upload depuis appareil photo
    camera_photo = st.camera_input("Ou prenez une photo")
    
    if camera_photo is not None:
        if camera_photo not in st.session_state.processed_images:
            st.session_state.processed_images.append(camera_photo)
    
    st.markdown("---")
    
    # Bouton pour traiter toutes les photos
    process_button = st.button(
        "🔍 Extraire les numéros", 
        type="primary", 
        use_container_width=True
    )
    
    # Bouton pour réinitialiser
    if st.button("🗑️ Tout effacer", use_container_width=True):
        st.session_state.processed_images = []
        st.session_state.all_results = []
        st.rerun()

# Zone principale
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Photos à traiter")
    
    # Ajouter les fichiers uploadés
    if uploaded_files:
        for file in uploaded_files:
            if file not in st.session_state.processed_images:
                st.session_state.processed_images.append(file)
    
    # Afficher les photos
    if st.session_state.processed_images:
        for idx, img_file in enumerate(st.session_state.processed_images):
            col_img, col_info, col_del = st.columns([2, 2, 1])
            
            with col_img:
                image = Image.open(img_file)
                st.image(image, caption=f"Photo {idx+1}", width=150)
            
            with col_info:
                st.write(f"**Fichier:** {img_file.name}")
                st.write(f"**Taille:** {image.size}")
            
            with col_del:
                if st.button("❌", key=f"del_{idx}"):
                    st.session_state.processed_images.pop(idx)
                    st.rerun()
    else:
        st.info("👆 Utilisez la barre latérale pour ajouter des photos")

with col2:
    st.subheader("📊 Résultats de l'extraction")
    
    if process_button and st.session_state.processed_images:
        # Traiter chaque image
        st.session_state.all_results = []
        progress_bar = st.progress(0)
        
        for idx, img_file in enumerate(st.session_state.processed_images):
            with st.spinner(f"Traitement de la photo {idx+1}..."):
                image = Image.open(img_file)
                result = process_image(image)
                
                # Ajouter les métadonnées
                result['filename'] = img_file.name
                result['processed_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                st.session_state.all_results.append(result)
                
            progress_bar.progress((idx + 1) / len(st.session_state.processed_images))
        
        st.success(f"✅ {len(st.session_state.all_results)} photos traitées avec succès!")
    
    # Afficher les résultats
    if st.session_state.all_results:
        # Créer un DataFrame pour l'affichage
        df_display = pd.DataFrame([
            {
                'Photo': r['filename'],
                'Nombre de numéros': r['count'],
                'Numéros trouvés': r['all_numbers'] if r['all_numbers'] else 'Aucun',
                'Texte complet': r['text'][:100] + '...' if len(r['text']) > 100 else r['text']
            }
            for r in st.session_state.all_results
        ])
        
        st.dataframe(df_display, use_container_width=True)
        
        # Détails pour chaque photo
        with st.expander("🔍 Voir les détails par photo"):
            for idx, result in enumerate(st.session_state.all_results):
                st.write(f"**Photo {idx+1}: {result['filename']}**")
                if result['success']:
                    st.write(f"📝 Texte extrait: {result['text']}")
                    st.write(f"🔢 Numéros: {', '.join(result['numbers']) if result['numbers'] else 'Aucun'}")
                else:
                    st.error(f"Erreur: {result['error']}")
                st.markdown("---")
        
        # Bouton de téléchargement Excel
        st.markdown("### 💾 Sauvegarde des résultats")
        
        # Préparer les données pour Excel
        excel_data = []
        for result in st.session_state.all_results:
            excel_data.append({
                'Nom du fichier': result['filename'],
                'Date de traitement': result['processed_date'],
                'Nombre de numéros': result['count'],
                'Numéros trouvés': result['all_numbers'],
                'Texte complet': result['text']
            })
        
        excel_file = save_to_excel(excel_data)
        
        st.download_button(
            label="📥 Télécharger les résultats (Excel)",
            data=excel_file,
            file_name=f"numeros_extraits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("### 📝 Instructions")
st.markdown("""
1. **Ajoutez des photos** via la barre latérale (upload ou appareil photo)
2. **Cliquez sur "Extraire les numéros"** pour lancer l'OCR
3. **Visualisez les résultats** dans le tableau
4. **Téléchargez le fichier Excel** avec tous les numéros extraits
""")
