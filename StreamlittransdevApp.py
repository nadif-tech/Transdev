import streamlit as st
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import io
import re
from datetime import datetime
import base64

# Configuration de la page
st.set_page_config(
    page_title="Extracteur de Numéros",
    page_icon="📸",
    layout="wide"
)

st.title("📸 Extracteur de Numéros depuis Photos")
st.markdown("---")

def preprocess_image(image):
    """Prétraitement simple de l'image"""
    if image.mode != 'L':
        image = image.convert('L')
    
    # Améliorer le contraste
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.5)
    
    # Améliorer la netteté
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    return image

def extract_numbers_from_filename(filename):
    """Extrait les numéros du nom de fichier"""
    numbers = re.findall(r'\b\d+\b', filename)
    return numbers

def manual_number_input(image, idx, filename):
    """Interface de saisie manuelle des numéros"""
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption=f"Photo {idx+1}: {filename}", width=300)
    
    with col2:
        st.write("### 📝 Saisie des numéros visibles")
        st.write("Entrez les numéros que vous voyez sur la photo:")
        
        # Exemple pour TESTO01.jpeg
        if "TESTO01" in filename.upper():
            st.info("💡 Exemple de numéros à entrer: 10406871, 823743")
        
        manual_text = st.text_area(
            "Numéros (un par ligne ou séparés par des virgules)",
            key=f"manual_input_{idx}",
            height=120,
            placeholder="Exemple:\n10406871\n823743"
        )
        
        if manual_text:
            # Extraire tous les nombres
            numbers = re.findall(r'\b\d+\b', manual_text)
            if numbers:
                st.success(f"✅ {len(numbers)} numéros détectés: {', '.join(numbers)}")
                
                # Option pour sauvegarder
                if st.button(f"💾 Sauvegarder ces numéros", key=f"save_{idx}"):
                    return numbers
            else:
                st.warning("⚠️ Aucun numéro détecté dans le texte")
        
        # Suggestion de numéros depuis le nom de fichier
        file_numbers = extract_numbers_from_filename(filename)
        if file_numbers:
            st.info(f"📎 Numéros trouvés dans le nom: {', '.join(file_numbers)}")
            if st.button(f"📎 Utiliser ces numéros", key=f"use_file_{idx}"):
                return file_numbers
    
    return []

def create_download_link(df, filename):
    """Crée un lien de téléchargement pour le CSV"""
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Télécharger le fichier CSV</a>'
    return href

def save_to_csv(data):
    """Convertit les données en CSV"""
    df = pd.DataFrame(data)
    return df.to_csv(index=False, encoding='utf-8-sig')

# Initialisation de la session state
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = {}

# Sidebar
with st.sidebar:
    st.header("⚙️ Options")
    
    st.markdown("### 📤 Upload de photos")
    
    uploaded_files = st.file_uploader(
        "Choisissez vos photos",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'],
        accept_multiple_files=True
    )
    
    st.markdown("### 📷 Appareil photo")
    camera_photo = st.camera_input("Prenez une photo")
    
    if camera_photo is not None:
        if camera_photo not in st.session_state.processed_images:
            st.session_state.processed_images.append(camera_photo)
    
    st.markdown("---")
    
    if st.button("🗑️ Tout effacer", use_container_width=True):
        st.session_state.processed_images = []
        st.session_state.extracted_data = {}
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    st.write(f"Photos chargées: {len(st.session_state.processed_images)}")
    st.write(f"Données extraites: {len(st.session_state.extracted_data)}")

# Zone principale
st.subheader("📋 Photos chargées")

# Ajouter les fichiers uploadés
if uploaded_files:
    for file in uploaded_files:
        if file not in st.session_state.processed_images:
            st.session_state.processed_images.append(file)

if st.session_state.processed_images:
    # Afficher les photos
    cols = st.columns(min(3, len(st.session_state.processed_images)))
    for idx, img_file in enumerate(st.session_state.processed_images):
        with cols[idx % 3]:
            image = Image.open(img_file)
            st.image(image, caption=f"{img_file.name}", use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"📄 {img_file.name}")
            with col2:
                if st.button("❌ Supprimer", key=f"del_{idx}"):
                    st.session_state.processed_images.pop(idx)
                    st.rerun()
    
    st.markdown("---")
    st.subheader("🔍 Extraction des numéros")
    
    # Interface d'extraction pour chaque photo
    for idx, img_file in enumerate(st.session_state.processed_images):
        image = Image.open(img_file)
        
        st.markdown(f"### 📸 Photo {idx+1}: {img_file.name}")
        
        # Afficher la photo
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, width=350)
        
        with col2:
            st.write("**Numéros extraits:**")
            
            # Vérifier si déjà extrait
            if img_file.name in st.session_state.extracted_data:
                numbers = st.session_state.extracted_data[img_file.name]
                st.success(f"✅ {len(numbers)} numéros enregistrés")
                st.code('\n'.join(numbers))
                
                if st.button(f"✏️ Modifier", key=f"edit_{idx}"):
                    del st.session_state.extracted_data[img_file.name]
                    st.rerun()
            else:
                # Saisie manuelle
                manual_text = st.text_area(
                    "Entrez les numéros visibles:",
                    key=f"text_{idx}",
                    height=100,
                    placeholder="10406871\n823743\nHENGSTLER"
                )
                
                # Extraire automatiquement du nom
                file_nums = extract_numbers_from_filename(img_file.name)
                if file_nums:
                    st.info(f"📎 Numéros dans le nom: {', '.join(file_nums)}")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button(f"💾 Sauvegarder", key=f"save_btn_{idx}"):
                        numbers = re.findall(r'\b\d+\b', manual_text)
                        if numbers:
                            st.session_state.extracted_data[img_file.name] = numbers
                            st.success(f"✅ {len(numbers)} numéros sauvegardés!")
                            st.rerun()
                        else:
                            st.error("❌ Aucun numéro trouvé")
                
                with col_b:
                    if file_nums and st.button(f"📎 Utiliser nom", key=f"use_file_btn_{idx}"):
                        st.session_state.extracted_data[img_file.name] = file_nums
                        st.success(f"✅ Numéros du fichier sauvegardés!")
                        st.rerun()
        
        st.markdown("---")
    
    # Affichage des résultats
    if st.session_state.extracted_data:
        st.subheader("📊 Résultats extraits")
        
        # Créer un DataFrame
        data_rows = []
        for filename, numbers in st.session_state.extracted_data.items():
            data_rows.append({
                'Fichier': filename,
                'Nombre de numéros': len(numbers),
                'Numéros': ', '.join(numbers),
                'Date extraction': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        df = pd.DataFrame(data_rows)
        st.dataframe(df, use_container_width=True)
        
        # Téléchargement CSV
        st.markdown("### 💾 Téléchargement")
        
        csv_data = save_to_csv(data_rows)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv_data,
                file_name=f"numeros_extraits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("📋 Copier tous les numéros", use_container_width=True):
                all_numbers = []
                for numbers in st.session_state.extracted_data.values():
                    all_numbers.extend(numbers)
                nums_text = ', '.join(all_numbers)
                st.code(nums_text)
                st.success("✅ Numéros copiés dans le presse-papier (utilisez Ctrl+C)")
    
else:
    st.info("👆 Utilisez la barre latérale pour ajouter des photos")
    
    # Mode démo
    st.markdown("### 🎯 Mode démonstration")
    st.write("Testez avec l'exemple TESTO01.jpeg:")
    
    demo_numbers = st.text_area(
        "Entrez les numéros de test:",
        value="10406871\n823743",
        height=100
    )
    
    if st.button("🧪 Tester l'extraction"):
        numbers = re.findall(r'\b\d+\b', demo_numbers)
        if numbers:
            st.success(f"✅ {len(numbers)} numéros trouvés!")
            
            # Créer un mini rapport
            demo_data = [{
                'Fichier': 'TESTO01.jpeg',
                'Numéros': ', '.join(numbers),
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }]
            
            df_demo = pd.DataFrame(demo_data)
            st.dataframe(df_demo)
            
            csv_demo = save_to_csv(demo_data)
            st.download_button(
                label="📥 Télécharger CSV (test)",
                data=csv_demo,
                file_name="test_numeros.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <h4>📌 Comment utiliser pour TESTO01.jpeg :</h4>
    <ol style='text-align: left; display: inline-block;'>
        <li>Chargez la photo via l'upload</li>
        <li>Entrez les numéros visibles: <b>10406871</b> et <b>823743</b></li>
        <li>Cliquez sur "Sauvegarder"</li>
        <li>Téléchargez le fichier CSV avec tous les numéros</li>
    </ol>
    <p style='margin-top: 20px;'>✅ 100% compatible Streamlit Cloud | Export CSV</p>
</div>
""", unsafe_allow_html=True)
