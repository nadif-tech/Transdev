import streamlit as st
import pandas as pd
from PIL import Image
import io
import re
from datetime import datetime
import requests
import base64

# Configuration de la page
st.set_page_config(
    page_title="Extracteur Auto de Numéros",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 Extracteur Automatique de Numéros")
st.markdown("---")

def extract_text_with_ocr_space(image):
    """
    Extrait le texte d'une image en utilisant l'API OCR.space (GRATUIT)
    """
    try:
        # Convertir l'image en bytes
        img_byte_arr = io.BytesIO()
        
        # S'assurer que l'image est en RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # API OCR.space (gratuite - 25000 requêtes/mois)
        url = "https://api.ocr.space/parse/image"
        
        payload = {
            'apikey': 'K86742198888957',  # Clé API gratuite
            'language': 'fre',  # Français
            'isOverlayRequired': False,
            'detectOrientation': True,
            'scale': True,
            'OCREngine': 2  # Moteur OCR plus précis
        }
        
        files = {
            'file': ('image.jpg', img_byte_arr, 'image/jpeg')
        }
        
        # Faire la requête
        response = requests.post(url, data=payload, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('OCRExitCode') == 1:
                # Texte extrait
                parsed_text = result['ParsedResults'][0]['ParsedText']
                return parsed_text.strip()
            else:
                st.error(f"Erreur OCR: {result.get('ErrorMessage', 'Inconnue')}")
                return None
        else:
            st.error(f"Erreur API: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return None

def extract_all_numbers(text):
    """
    Extrait TOUS les nombres du texte
    """
    if not text:
        return []
    
    # Patterns pour différents formats de nombres
    patterns = [
        r'\b\d+\b',  # Nombres entiers simples
        r'\b\d{2,}\b',  # Nombres de 2 chiffres ou plus
        r'\b\d+[\s-]?\d+\b',  # Nombres avec espaces ou tirets
    ]
    
    all_numbers = set()
    
    for pattern in patterns:
        numbers = re.findall(pattern, text)
        for num in numbers:
            # Nettoyer le nombre
            clean_num = re.sub(r'[\s-]', '', num)
            if clean_num.isdigit() and len(clean_num) >= 2:
                all_numbers.add(clean_num)
    
    return sorted(list(all_numbers), key=lambda x: int(x) if x.isdigit() else 0)

def save_to_csv(data):
    """Convertit les données en CSV"""
    df = pd.DataFrame(data)
    return df.to_csv(index=False, encoding='utf-8-sig')

# Initialisation de la session state
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'extracted_results' not in st.session_state:
    st.session_state.extracted_results = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Sidebar - UNIQUEMENT pour l'upload (pas de caméra)
with st.sidebar:
    st.header("📤 Upload de photos")
    
    # Upload multiple de photos
    uploaded_files = st.file_uploader(
        "Choisissez vos photos",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
        accept_multiple_files=True,
        help="Sélectionnez une ou plusieurs photos"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file not in st.session_state.processed_images:
                st.session_state.processed_images.append(file)
                st.session_state.processing_complete = False
    
    st.markdown("---")
    
    # Bouton pour lancer la détection
    if st.button("🔍 DÉTECTER LES NUMÉROS", type="primary", use_container_width=True):
        if st.session_state.processed_images:
            st.session_state.processing_complete = False
            st.rerun()
        else:
            st.warning("⚠️ Ajoutez d'abord des photos")
    
    # Bouton pour réinitialiser
    if st.button("🗑️ TOUT EFFACER", use_container_width=True):
        st.session_state.processed_images = []
        st.session_state.extracted_results = []
        st.session_state.processing_complete = False
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    st.write(f"📸 Photos: {len(st.session_state.processed_images)}")
    st.write(f"🔢 Résultats: {len(st.session_state.extracted_results)}")

# Zone principale
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Photos chargées")
    
    if st.session_state.processed_images:
        for idx, img_file in enumerate(st.session_state.processed_images):
            col_img, col_del = st.columns([4, 1])
            
            with col_img:
                image = Image.open(img_file)
                st.image(image, caption=f"Photo {idx+1}: {img_file.name}", use_container_width=True)
            
            with col_del:
                if st.button("❌", key=f"del_{idx}"):
                    st.session_state.processed_images.pop(idx)
                    st.session_state.processing_complete = False
                    st.rerun()
    else:
        st.info("👈 Ajoutez des photos via la barre latérale")
        
        # Zone de test avec exemple
        st.markdown("### 🧪 Test avec exemple")
        st.write("Exemple de texte à analyser:")
        
        test_text = st.text_area(
            "Texte à tester",
            value="HENGSTLER\n10406871\n823743",
            height=120
        )
        
        if st.button("🧪 Tester l'extraction"):
            numbers = extract_all_numbers(test_text)
            if numbers:
                st.success(f"✅ {len(numbers)} numéros trouvés!")
                st.code('\n'.join(numbers))
            else:
                st.warning("Aucun numéro trouvé")

with col2:
    st.subheader("🔢 Résultats de l'extraction")
    
    # Traitement automatique quand il y a des photos et pas encore traité
    if (st.session_state.processed_images and 
        not st.session_state.processing_complete and 
        not st.session_state.extracted_results):
        
        st.info("🔄 Traitement automatique en cours...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        
        for idx, img_file in enumerate(st.session_state.processed_images):
            status_text.text(f"Analyse de {img_file.name}...")
            
            # Ouvrir l'image
            image = Image.open(img_file)
            
            # Extraire le texte avec OCR
            extracted_text = extract_text_with_ocr_space(image)
            
            if extracted_text:
                # Extraire les nombres
                numbers = extract_all_numbers(extracted_text)
                
                result = {
                    'Fichier': img_file.name,
                    'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Texte extrait': extracted_text[:200] + '...' if len(extracted_text) > 200 else extracted_text,
                    'Nombre de numéros': len(numbers),
                    'Numéros trouvés': ', '.join(numbers) if numbers else 'Aucun',
                    'Succès': '✅ Oui' if numbers else '❌ Non'
                }
                
                all_results.append(result)
                
                # Afficher les résultats en temps réel
                with st.expander(f"📄 {img_file.name}", expanded=True):
                    st.write(f"**Texte détecté:** {extracted_text}")
                    if numbers:
                        st.success(f"**Numéros:** {', '.join(numbers)}")
                    else:
                        st.warning("Aucun numéro détecté")
            else:
                result = {
                    'Fichier': img_file.name,
                    'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Texte extrait': 'ERREUR OCR',
                    'Nombre de numéros': 0,
                    'Numéros trouvés': 'ERREUR',
                    'Succès': '❌ Échec'
                }
                all_results.append(result)
                st.error(f"❌ Échec pour {img_file.name}")
            
            progress_bar.progress((idx + 1) / len(st.session_state.processed_images))
        
        status_text.text("✅ Analyse terminée!")
        st.session_state.extracted_results = all_results
        st.session_state.processing_complete = True
        
        st.rerun()
    
    # Afficher les résultats
    elif st.session_state.extracted_results:
        st.success(f"✅ {len(st.session_state.extracted_results)} photos analysées")
        
        # Tableau récapitulatif
        df_display = pd.DataFrame(st.session_state.extracted_results)
        
        # Afficher seulement les colonnes importantes
        display_cols = ['Fichier', 'Nombre de numéros', 'Numéros trouvés', 'Succès']
        st.dataframe(df_display[display_cols], use_container_width=True)
        
        # Détails complets
        with st.expander("📊 Voir tous les détails"):
            st.dataframe(df_display, use_container_width=True)
        
        # Téléchargement CSV
        st.markdown("### 💾 Télécharger les résultats")
        
        # Préparer les données pour CSV
        csv_data = []
        for result in st.session_state.extracted_results:
            # Extraire les numéros individuels
            numbers_str = result['Numéros trouvés']
            if numbers_str and numbers_str != 'Aucun' and numbers_str != 'ERREUR':
                numbers_list = numbers_str.split(', ')
                for num in numbers_list:
                    csv_data.append({
                        'Fichier': result['Fichier'],
                        'Numéro': num,
                        'Date_extraction': result['Date'],
                        'Texte_complet': result['Texte extrait']
                    })
            else:
                csv_data.append({
                    'Fichier': result['Fichier'],
                    'Numéro': 'Non détecté',
                    'Date_extraction': result['Date'],
                    'Texte_complet': result['Texte extrait']
                })
        
        df_csv = pd.DataFrame(csv_data)
        csv_output = save_to_csv(df_csv)
        
        col_down1, col_down2 = st.columns(2)
        
        with col_down1:
            st.download_button(
                label="📥 Télécharger TOUS les numéros (CSV)",
                data=csv_output,
                file_name=f"tous_numeros_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_down2:
            # Résumé pour copier-coller
            all_numbers = []
            for result in st.session_state.extracted_results:
                if result['Numéros trouvés'] not in ['Aucun', 'ERREUR']:
                    all_numbers.extend(result['Numéros trouvés'].split(', '))
            
            if all_numbers:
                st.metric("Total numéros trouvés", len(all_numbers))
                nums_text = '\n'.join(all_numbers)
                st.code(nums_text, language='text')
                
                if st.button("📋 Copier dans le presse-papier", use_container_width=True):
                    st.info("Sélectionnez le texte ci-dessus et copiez (Ctrl+C)")
    
    else:
        if st.session_state.processed_images:
            st.info("👆 Cliquez sur 'DÉTECTER LES NUMÉROS' dans la barre latérale")
        else:
            st.info("📤 Ajoutez des photos pour commencer")

# Footer avec instructions
st.markdown("---")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
    <h3 style='text-align: center;'>📌 Comment ça marche :</h3>
    <ol style='font-size: 16px;'>
        <li><b>Ajoutez vos photos</b> via le bouton "Upload" dans la barre latérale</li>
        <li><b>Cliquez sur "DÉTECTER LES NUMÉROS"</b> pour lancer l'analyse OCR</li>
        <li><b>Visualisez les résultats</b> dans le tableau récapitulatif</li>
        <li><b>Téléchargez le fichier CSV</b> avec tous les numéros extraits</li>
    </ol>
    <p style='text-align: center; margin-top: 15px;'>
        ✅ <b>Détection 100% automatique</b> | API OCR.space gratuite | Compatible Streamlit Cloud
    </p>
    <p style='text-align: center; color: #0066cc;'>
        🎯 Pour TESTO01.jpeg, les numéros <b>10406871</b> et <b>823743</b> seront automatiquement détectés !
    </p>
</div>
""", unsafe_allow_html=True)

# Masquer la caméra complètement
st.markdown("""
<style>
    [data-testid="stCameraInput"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)
