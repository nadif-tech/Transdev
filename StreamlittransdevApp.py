import streamlit as st
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
from datetime import datetime

st.set_page_config(
    page_title="Extracteur Numeros Compteur",
    page_icon="123",
    layout="wide"
)

st.title("Extracteur de Numeros - Compteur Horaire/Kilometrage")
st.markdown("---")

def preprocess_image(image):
    """Pretraitement de l'image"""
    if image.mode != 'L':
        image = image.convert('L')
    
    width, height = image.size
    if width < 1200:
        ratio = 1200 / width
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(3.0)
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.5)
    
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    return image

def find_text_regions(binary_image):
    """Trouve les regions de texte"""
    width, height = binary_image.size
    pixels = binary_image.load()
    
    visited = [[False for _ in range(width)] for _ in range(height)]
    regions = []
    
    for y in range(height):
        for x in range(width):
            if pixels[x, y] == 0 and not visited[y][x]:
                component = []
                queue = [(x, y)]
                visited[y][x] = True
                
                min_x = max_x = x
                min_y = max_y = y
                
                while queue:
                    cx, cy = queue.pop(0)
                    component.append((cx, cy))
                    
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)
                    
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = cx + dx, cy + dy
                            if (0 <= nx < width and 0 <= ny < height and
                                not visited[ny][nx] and pixels[nx, ny] == 0):
                                queue.append((nx, ny))
                                visited[ny][nx] = True
                
                if len(component) > 30:
                    regions.append({
                        'bbox': (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
                        'size': len(component),
                        'center_y': (min_y + max_y) // 2
                    })
    
    return regions

def group_into_lines(regions):
    """Groupe les regions en lignes"""
    if not regions:
        return []
    
    sorted_regions = sorted(regions, key=lambda r: r['center_y'])
    lines = []
    current_line = [sorted_regions[0]]
    
    for i in range(1, len(sorted_regions)):
        if abs(sorted_regions[i]['center_y'] - sorted_regions[i-1]['center_y']) <= 20:
            current_line.append(sorted_regions[i])
        else:
            lines.append(current_line)
            current_line = [sorted_regions[i]]
    
    lines.append(current_line)
    return lines

def extract_number_from_line(line_regions):
    """Extrait un numero d'une ligne"""
    if len(line_regions) < 3:
        return None
    
    sorted_line = sorted(line_regions, key=lambda r: r['bbox'][0])
    
    min_x = min(r['bbox'][0] for r in sorted_line)
    max_x = max(r['bbox'][0] + r['bbox'][2] for r in sorted_line)
    
    line_width = max_x - min_x
    estimated_chars = line_width // 12
    
    if estimated_chars >= 4:
        return estimated_chars
    
    return None

def analyze_image(image):
    """Analyse complete de l'image"""
    try:
        processed = preprocess_image(image)
        
        thresholds = [140, 160, 180, 200]
        all_numbers = []
        
        for thresh in thresholds:
            binary = processed.point(lambda x: 0 if x < thresh else 255, '1')
            regions = find_text_regions(binary)
            lines = group_into_lines(regions)
            
            for line in lines:
                chars = extract_number_from_line(line)
                if chars:
                    all_numbers.append(chars)
        
        if all_numbers:
            max_chars = max(all_numbers)
            min_chars = min(all_numbers)
            
            result = {
                'heures': [f"NUMERO_{max_chars}_CHIFFRES"],
                'kilometrages': [f"NUMERO_{min_chars}_CHIFFRES"],
                'autres': [],
                'regions_detectees': len(regions),
                'lignes_detectees': len(lines)
            }
        else:
            result = {
                'heures': [],
                'kilometrages': [],
                'autres': [],
                'regions_detectees': len(regions),
                'lignes_detectees': len(lines)
            }
        
        return result
        
    except Exception as e:
        return {
            'heures': [],
            'kilometrages': [],
            'autres': [],
            'error': str(e)
        }

def manual_extraction_interface(image, idx):
    """Interface de saisie manuelle"""
    st.image(image, caption=f"Photo {idx+1}", width=300)
    
    col1, col2 = st.columns(2)
    
    with col1:
        heures = st.text_input(
            "Nombre d'heures",
            key=f"heures_{idx}",
            placeholder="Ex: 10406871"
        )
    
    with col2:
        km = st.text_input(
            "Kilometrage",
            key=f"km_{idx}",
            placeholder="Ex: 623743"
        )
    
    return heures, km

if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'results' not in st.session_state:
    st.session_state.results = []
if 'mode' not in st.session_state:
    st.session_state.mode = "Manuel"

with st.sidebar:
    st.header("Configuration")
    
    st.session_state.mode = st.radio(
        "Mode d'extraction",
        ["Manuel", "Automatique"],
        help="Manuel: saisie directe\nAutomatique: detection par analyse d'image"
    )
    
    st.markdown("---")
    
    st.header("Upload Photos")
    
    uploaded_files = st.file_uploader(
        "Choisissez vos photos",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.session_state.uploaded_images = uploaded_files
        st.success(f"{len(uploaded_files)} photo(s)")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("VALIDER", type="primary", use_container_width=True):
            if not st.session_state.uploaded_images:
                st.warning("Chargez des photos")
            else:
                st.session_state.results = []
                st.rerun()
    
    with col2:
        if st.button("EFFACER", use_container_width=True):
            st.session_state.uploaded_images = []
            st.session_state.results = []
            st.rerun()

st.header("Extraction des donnees")

if st.session_state.uploaded_images:
    if st.session_state.mode == "Manuel":
        st.subheader("Saisie manuelle des numeros")
        
        all_results = []
        
        for idx, img_file in enumerate(st.session_state.uploaded_images):
            st.markdown(f"### Photo {idx+1}: {img_file.name}")
            
            image = Image.open(img_file)
            heures, km = manual_extraction_interface(image, idx)
            
            if heures or km:
                result = {
                    'filename': img_file.name,
                    'heures': [heures] if heures else [],
                    'kilometrages': [km] if km else [],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                all_results.append(result)
            
            st.markdown("---")
        
        if st.button("ENREGISTRER TOUT", type="primary"):
            st.session_state.results = all_results
            st.success(f"{len(all_results)} photos enregistrees!")
            st.rerun()
    
    else:
        st.subheader("Analyse automatique")
        
        if st.button("LANCER L'ANALYSE", type="primary"):
            progress_bar = st.progress(0)
            
            for idx, img_file in enumerate(st.session_state.uploaded_images):
                st.write(f"Analyse: {img_file.name}")
                
                image = Image.open(img_file)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(image, width=250)
                
                with col2:
                    with st.spinner("Analyse..."):
                        result = analyze_image(image)
                        result['filename'] = img_file.name
                        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        if result['heures']:
                            st.success(f"Heures detectees: {', '.join(result['heures'])}")
                        if result['kilometrages']:
                            st.success(f"Km detectes: {', '.join(result['kilometrages'])}")
                        
                        st.session_state.results.append(result)
                
                progress_bar.progress((idx + 1) / len(st.session_state.uploaded_images))
            
            st.success("Analyse terminee!")
            st.rerun()

if st.session_state.results:
    st.markdown("---")
    st.subheader("Resultats enregistres")
    
    summary_data = []
    all_heures = []
    all_km = []
    
    for r in st.session_state.results:
        summary_data.append({
            'Fichier': r['filename'],
            'Heures': ', '.join(r['heures']) if r['heures'] else '-',
            'Kilometrage': ', '.join(r['kilometrages']) if r['kilometrages'] else '-',
            'Date': r['timestamp']
        })
        
        all_heures.extend(r['heures'])
        all_km.extend(r['kilometrages'])
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Export CSV")
    
    export_data = []
    for r in st.session_state.results:
        for h in r['heures']:
            export_data.append({'Fichier': r['filename'], 'Type': 'Heures', 'Valeur': h})
        for k in r['kilometrages']:
            export_data.append({'Fichier': r['filename'], 'Type': 'Kilometrage', 'Valeur': k})
    
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

st.markdown("---")
st.markdown("""
<div style='background: #1e3c72; padding: 20px; border-radius: 10px; color: white; text-align: center;'>
    <h3>Extracteur de Numeros - Compteurs</h3>
    <p>Mode Manuel + Automatique | Python Pur</p>
</div>
""", unsafe_allow_html=True)
