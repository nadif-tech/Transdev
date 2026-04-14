import streamlit as st
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import io
import re
from datetime import datetime
import math

# Configuration de la page
st.set_page_config(
    page_title="Extracteur de Numéros",
    page_icon="📸",
    layout="wide"
)

st.title("📸 Extracteur de Numéros depuis Photos")
st.markdown("---")

def preprocess_image_pil(image):
    """Prétraitement d'image avec PIL uniquement"""
    # Convertir en niveaux de gris
    if image.mode != 'L':
        image = image.convert('L')
    
    # Améliorer le contraste
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Améliorer la netteté
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    # Appliquer un filtre pour faire ressortir les bords
    image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    return image

def binarize_image(image, threshold=128):
    """Binarisation simple de l'image"""
    # Convertir en noir et blanc
    bw_image = image.point(lambda x: 0 if x < threshold else 255, '1')
    return bw_image

def find_connected_components(bw_image):
    """Trouve les composants connectés (chiffres potentiels)"""
    # Convertir en tableau de pixels
    pixels = bw_image.load()
    width, height = bw_image.size
    
    # Matrice pour marquer les pixels visités
    visited = [[False for _ in range(width)] for _ in range(height)]
    
    components = []
    
    # Parcourir tous les pixels
    for y in range(height):
        for x in range(width):
            if pixels[x, y] == 0 and not visited[y][x]:  # Pixel noir
                # BFS pour trouver le composant connecté
                component = []
                queue = [(x, y)]
                visited[y][x] = True
                
                min_x, max_x = x, x
                min_y, max_y = y, y
                
                while queue:
                    cx, cy = queue.pop(0)
                    component.append((cx, cy))
                    
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)
                    
                    # Vérifier les 8 voisins
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = cx + dx, cy + dy
                            if (0 <= nx < width and 0 <= ny < height and 
                                not visited[ny][nx] and pixels[nx, ny] == 0):
                                queue.append((nx, ny))
                                visited[ny][nx] = True
                
                if len(component) > 10:  # Ignorer le bruit
                    bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
                    components.append({
                        'bbox': bbox,
                        'size': len(component),
                        'pixels': component
                    })
    
    return components

def is_likely_digit(component):
    """Détermine si un composant ressemble à un chiffre"""
    bbox = component['bbox']
    width, height = bbox[2], bbox[3]
    
    # Critères pour un chiffre
    if width < 5 or height < 8:  # Trop petit
        return False
    
    if width > 100 or height > 200:  # Trop grand
        return False
    
    # Ratio hauteur/largeur typique pour les chiffres
    aspect_ratio = height / width
    if aspect_ratio < 1.2 or aspect_ratio > 4.0:
        return False
    
    # Densité de pixels
    density = component['size'] / (width * height)
    if density < 0.1 or density > 0.9:
        return False
    
    return True

def group_digits_into_numbers(components, max_gap=15, vertical_tolerance=10):
    """Groupe les chiffres en nombres basé sur la proximité"""
    if not components:
        return []
    
    # Trier par position x
    sorted_comps = sorted(components, key=lambda c: c['bbox'][0])
    
    numbers = []
    current_number = [sorted_comps[0]]
    
    for i in range(1, len(sorted_comps)):
        current = sorted_comps[i]
        previous = sorted_comps[i-1]
        
        # Distance horizontale
        x_distance = current['bbox'][0] - (previous['bbox'][0] + previous['bbox'][2])
        
        # Différence verticale
        y_center_current = current['bbox'][1] + current['bbox'][3] / 2
        y_center_previous = previous['bbox'][1] + previous['bbox'][3] / 2
        y_difference = abs(y_center_current - y_center_previous)
        
        # Si proche, ajouter au nombre courant
        if x_distance <= max_gap and y_difference <= vertical_tolerance:
            current_number.append(current)
        else:
            # Commencer un nouveau nombre
            if len(current_number) >= 1:
                numbers.append(current_number)
            current_number = [current]
    
    if current_number:
        numbers.append(current_number)
    
    return numbers

def recognize_digit_simple(component):
    """Reconnaissance simple de chiffres basée sur des caractéristiques"""
    bbox = component['bbox']
    width, height = bbox[2], bbox[3]
    density = component['size'] / (width * height)
    aspect_ratio = height / width
    
    # Logique simple de reconnaissance
    if aspect_ratio > 2.5 and density < 0.4:
        return '1'
    elif density > 0.7:
        return '8'
    elif density < 0.3:
        return '7'
    elif 1.5 < aspect_ratio < 2.0:
        return '4'
    else:
        return '0'  # Par défaut
    
    # Note: Cette reconnaissance est basique. Pour une meilleure précision,
    # il faudrait implémenter un OCR plus sophistiqué ou utiliser une API.

def extract_text_with_regex(image):
    """Extrait le texte en utilisant des patterns regex sur les métadonnées"""
    # Cette méthode est un fallback simple
    numbers = []
    
    # Extraire du nom de fichier si disponible
    if hasattr(image, 'filename') and image.filename:
        numbers.extend(re.findall(r'\b\d+\b', image.filename))
    
    # Vérifier les métadonnées EXIF
    if hasattr(image, '_getexif') and image._getexif():
        exif = image._getexif()
        if exif:
            for value in exif.values():
                if isinstance(value, str):
                    numbers.extend(re.findall(r'\b\d+\b', value))
    
    return numbers

def process_image_pure_python(image):
    """Traite une image avec du Python pur (PIL uniquement)"""
    try:
        # Prétraitement
        processed = preprocess_image_pil(image)
        
        # Binarisation
        binary = binarize_image(processed, threshold=150)
        
        # Trouver les composants
        components = find_connected_components(binary)
        
        # Filtrer les composants qui ressemblent à des chiffres
        digit_components = [c for c in components if is_likely_digit(c)]
        
        # Grouper en nombres
        number_groups = group_digits_into_numbers(digit_components)
        
        # Reconnaître les chiffres (version simple)
        recognized_numbers = []
        for group in number_groups:
            digits = []
            for comp in group:
                digit = recognize_digit_simple(comp)
                digits.append(digit)
            
            number_str = ''.join(digits)
            recognized_numbers.append(number_str)
        
        # Fallback: chercher avec regex
        regex_numbers = extract_text_with_regex(image)
        all_numbers = list(set(recognized_numbers + regex_numbers))
        
        # Filtrer les nombres valides
        valid_numbers = [n for n in all_numbers if len(n) >= 1 and n.isdigit()]
        
        return {
            'text': ' '.join(valid_numbers),
            'numbers': valid_numbers,
            'all_numbers': ', '.join(valid_numbers) if valid_numbers else '',
            'count': len(valid_numbers),
            'components_found': len(components),
            'digits_found': len(digit_components),
            'number_groups': len(number_groups),
            'success': True
        }
        
    except Exception as e:
        return {
            'text': '',
            'numbers': [],
            'all_numbers': '',
            'count': 0,
            'components_found': 0,
            'digits_found': 0,
            'number_groups': 0,
            'success': False,
            'error': str(e)
        }

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
    st.header("⚙️ Configuration")
    
    st.subheader("Paramètres de détection")
    
    threshold = st.slider(
        "Seuil de binarisation",
        min_value=50,
        max_value=200,
        value=150,
        step=10,
        help="Ajuste la sensibilité de détection"
    )
    
    min_size = st.slider(
        "Taille minimale (pixels)",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )
    
    st.markdown("---")
    
    # Upload de photos
    uploaded_files = st.file_uploader(
        "📁 Choisissez vos photos",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'],
        accept_multiple_files=True
    )
    
    # Appareil photo
    camera_photo = st.camera_input("📷 Prenez une photo")
    
    if camera_photo is not None:
        if camera_photo not in st.session_state.processed_images:
            st.session_state.processed_images.append(camera_photo)
    
    st.markdown("---")
    
    # Boutons d'action
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

# Zone principale
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Photos à traiter")
    
    if uploaded_files:
        for file in uploaded_files:
            if file not in st.session_state.processed_images:
                st.session_state.processed_images.append(file)
    
    if st.session_state.processed_images:
        for idx, img_file in enumerate(st.session_state.processed_images):
            col_img, col_info, col_del = st.columns([2, 2, 1])
            
            with col_img:
                image = Image.open(img_file)
                st.image(image, caption=f"Photo {idx+1}", width=150)
            
            with col_info:
                st.write(f"**{img_file.name}**")
                st.write(f"Taille: {image.size}")
            
            with col_del:
                if st.button("❌", key=f"del_{idx}"):
                    st.session_state.processed_images.pop(idx)
                    st.rerun()
    else:
        st.info("👆 Ajoutez des photos via la barre latérale")
        
        # Zone de démonstration
        st.markdown("### 📌 Comment ça marche:")
        st.markdown("""
        1. **Ajoutez** des photos (upload ou caméra)
        2. **Cliquez** sur 'Extraire les numéros'
        3. **Visualisez** les résultats
        4. **Téléchargez** en Excel
        
        ⚡ Utilise du Python pur - Aucune dépendance lourde!
        """)

with col2:
    st.subheader("📊 Résultats")
    
    if process_button and st.session_state.processed_images:
        st.session_state.all_results = []
        progress_bar = st.progress(0)
        
        for idx, img_file in enumerate(st.session_state.processed_images):
            with st.spinner(f"Traitement {idx+1}/{len(st.session_state.processed_images)}..."):
                image = Image.open(img_file)
                result = process_image_pure_python(image)
                
                result['filename'] = img_file.name
                result['processed_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result['image_size'] = f"{image.size[0]}x{image.size[1]}"
                
                st.session_state.all_results.append(result)
                
            progress_bar.progress((idx + 1) / len(st.session_state.processed_images))
        
        st.success(f"✅ {len(st.session_state.all_results)} photos traitées")
    
    if st.session_state.all_results:
        # Tableau des résultats
        df_display = pd.DataFrame([
            {
                'Photo': r['filename'],
                'Numéros': r['count'],
                'Valeurs': r['all_numbers'][:30] + ('...' if len(r['all_numbers']) > 30 else ''),
                'Composants': r['components_found'],
                'Chiffres': r['digits_found']
            }
            for r in st.session_state.all_results
        ])
        
        st.dataframe(df_display, use_container_width=True)
        
        # Détails
        with st.expander("🔍 Voir les détails"):
            for idx, result in enumerate(st.session_state.all_results):
                st.write(f"**{result['filename']}**")
                if result['success']:
                    st.write(f"✓ Composants: {result['components_found']}")
                    st.write(f"✓ Chiffres détectés: {result['digits_found']}")
                    st.write(f"✓ Groupes de nombres: {result['number_groups']}")
                    st.write(f"✓ Numéros: {result['all_numbers']}")
                else:
                    st.error(f"Erreur: {result.get('error', 'Inconnue')}")
                st.markdown("---")
        
        # Export Excel
        st.markdown("### 💾 Sauvegarde")
        
        excel_data = []
        for result in st.session_state.all_results:
            excel_data.append({
                'Fichier': result['filename'],
                'Date': result['processed_date'],
                'Taille': result['image_size'],
                'Numéros trouvés': result['count'],
                'Valeurs': result['all_numbers'],
                'Composants': result['components_found'],
                'Chiffres': result['digits_found']
            })
        
        excel_file = save_to_excel(excel_data)
        
        st.download_button(
            label="📥 Télécharger Excel",
            data=excel_file,
            file_name=f"numeros_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔍 Extracteur de Numéros | 100% Python Pur | Compatible Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)
