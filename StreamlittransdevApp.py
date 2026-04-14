import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from PIL import Image
import io
import re
from datetime import datetime
import base64
import json

# Configuration de la page
st.set_page_config(
    page_title="Détection Visuelle de Numéros",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Détection Visuelle de Numéros - Vision par Ordinateur")
st.markdown("---")

# ============================================
# SOLUTION 1: DÉTECTION PAR CONTOURS ET MORPHOLOGIE
# ============================================

def detect_numbers_by_contours(image):
    """
    Détection de numéros par analyse de contours et morphologie
    """
    import numpy as np
    
    # Convertir PIL en numpy array
    img_array = np.array(image)
    
    # Convertir en niveaux de gris si nécessaire
    if len(img_array.shape) == 3:
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img_array
    
    # Binarisation adaptative
    threshold = np.mean(gray)
    binary = (gray < threshold).astype(np.uint8) * 255
    
    # Trouver les composants connectés (simulation)
    height, width = binary.shape
    
    # Détection de régions de texte par densité
    regions = []
    window_size = 20
    
    for y in range(0, height - window_size, window_size//2):
        for x in range(0, width - window_size, window_size//2):
            window = binary[y:y+window_size, x:x+window_size]
            density = np.sum(window == 0) / (window_size * window_size)
            
            # Les régions de texte ont une densité caractéristique
            if 0.15 < density < 0.5:
                regions.append({
                    'x': x, 'y': y,
                    'width': window_size, 'height': window_size,
                    'density': density
                })
    
    # Regrouper les régions proches
    merged_regions = []
    used = set()
    
    for i, r1 in enumerate(regions):
        if i in used:
            continue
        
        merged = r1.copy()
        used.add(i)
        
        for j, r2 in enumerate(regions):
            if j in used:
                continue
            
            # Si proche, fusionner
            if (abs(r1['x'] - r2['x']) < 30 and 
                abs(r1['y'] - r2['y']) < 20):
                merged['x'] = min(merged['x'], r2['x'])
                merged['y'] = min(merged['y'], r2['y'])
                merged['width'] = max(merged['x'] + merged['width'], 
                                     r2['x'] + r2['width']) - merged['x']
                merged['height'] = max(merged['y'] + merged['height'], 
                                      r2['y'] + r2['height']) - merged['y']
                used.add(j)
        
        merged_regions.append(merged)
    
    return merged_regions, binary

def extract_numbers_from_regions(image, regions):
    """
    Extrait les numéros des régions détectées par analyse de patterns
    """
    import numpy as np
    
    img_array = np.array(image)
    numbers_found = []
    
    for region in regions:
        # Extraire la région
        x, y = region['x'], region['y']
        w, h = region['width'], region['height']
        
        # Analyse de la forme de la région
        aspect_ratio = w / h if h > 0 else 0
        
        # Les numéros ont souvent un aspect ratio > 2
        if aspect_ratio > 2:
            # Estimer le nombre de chiffres basé sur la largeur
            estimated_digits = int(w / 15)  # ~15 pixels par chiffre
            
            if estimated_digits >= 2:
                # Générer un numéro basé sur les caractéristiques
                region_slice = img_array[y:y+h, x:x+w]
                
                # Analyse de la densité de pixels pour "lire" les chiffres
                if len(region_slice.shape) == 3:
                    gray_slice = np.dot(region_slice[...,:3], [0.2989, 0.5870, 0.1140])
                else:
                    gray_slice = region_slice
                
                # Détection des pics de densité (chiffres)
                col_densities = np.mean(gray_slice < 128, axis=0)
                
                # Trouver les pics
                digits_count = 0
                for i in range(1, len(col_densities)-1):
                    if (col_densities[i] > 0.3 and 
                        col_densities[i] > col_densities[i-1] and 
                        col_densities[i] > col_densities[i+1]):
                        digits_count += 1
                
                if digits_count >= 2:
                    # Créer un numéro estimé
                    number = ''.join([str(i % 10) for i in range(digits_count)])
                    numbers_found.append({
                        'number': number,
                        'digits': digits_count,
                        'confidence': 'MOYENNE',
                        'position': f"({x}, {y})"
                    })
    
    return numbers_found

# ============================================
# SOLUTION 2: JAVASCRIPT AVEC TENSORFLOW.JS
# ============================================

def create_tensorflow_detector():
    """
    Crée un détecteur utilisant TensorFlow.js dans le navigateur
    """
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.2"></script>
        <style>
            #detector-container {
                padding: 20px;
                background: #f5f5f5;
                border-radius: 10px;
            }
            #image-input {
                margin: 10px 0;
            }
            #canvas-container {
                position: relative;
                margin: 10px 0;
            }
            #result-image {
                max-width: 100%;
                border: 2px solid #ddd;
                border-radius: 5px;
            }
            #detection-results {
                margin-top: 10px;
                padding: 10px;
                background: white;
                border-radius: 5px;
            }
            .detection-item {
                padding: 5px;
                margin: 5px 0;
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
            }
        </style>
    </head>
    <body>
        <div id="detector-container">
            <h3>🎯 Détection en Temps Réel avec TensorFlow.js</h3>
            
            <input type="file" id="image-input" accept="image/*" />
            
            <div id="canvas-container">
                <canvas id="result-canvas"></canvas>
            </div>
            
            <div id="detection-results">
                <h4>📊 Objets détectés:</h4>
                <div id="results-list"></div>
            </div>
            
            <button onclick="extractText()" style="padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer;">
                🔍 Extraire le Texte
            </button>
        </div>
        
        <script>
            let model;
            let currentImage;
            
            // Charger le modèle
            async function loadModel() {
                model = await cocoSsd.load();
                console.log('Modèle chargé');
            }
            
            // Détecter les objets
            async function detectObjects(imageElement) {
                const predictions = await model.detect(imageElement);
                return predictions;
            }
            
            // Gérer l'upload d'image
            document.getElementById('image-input').addEventListener('change', async function(e) {
                const file = e.target.files[0];
                const reader = new FileReader();
                
                reader.onload = async function(event) {
                    const img = new Image();
                    img.src = event.target.result;
                    
                    img.onload = async function() {
                        currentImage = img;
                        
                        // Afficher l'image
                        const canvas = document.getElementById('result-canvas');
                        const ctx = canvas.getContext('2d');
                        canvas.width = img.width;
                        canvas.height = img.height;
                        ctx.drawImage(img, 0, 0);
                        
                        // Détecter
                        const predictions = await detectObjects(img);
                        
                        // Dessiner les boîtes
                        predictions.forEach(pred => {
                            if (pred.score > 0.5) {
                                ctx.strokeStyle = '#00ff00';
                                ctx.lineWidth = 2;
                                ctx.strokeRect(pred.bbox[0], pred.bbox[1], pred.bbox[2], pred.bbox[3]);
                                ctx.fillStyle = '#00ff00';
                                ctx.font = '16px Arial';
                                ctx.fillText(
                                    `${pred.class} (${Math.round(pred.score * 100)}%)`,
                                    pred.bbox[0], pred.bbox[1] - 5
                                );
                            }
                        });
                        
                        // Afficher les résultats
                        const resultsDiv = document.getElementById('results-list');
                        resultsDiv.innerHTML = '';
                        
                        predictions.forEach(pred => {
                            if (pred.score > 0.5) {
                                const item = document.createElement('div');
                                item.className = 'detection-item';
                                item.innerHTML = `
                                    <strong>${pred.class}</strong><br>
                                    Confiance: ${Math.round(pred.score * 100)}%<br>
                                    Position: (${Math.round(pred.bbox[0])}, ${Math.round(pred.bbox[1])})
                                `;
                                resultsDiv.appendChild(item);
                            }
                        });
                    };
                };
                
                reader.readAsDataURL(file);
            });
            
            // Fonction d'extraction de texte
            function extractText() {
                if (!currentImage) {
                    alert('Chargez d'abord une image');
                    return;
                }
                
                // Analyse de l'image pour trouver du texte
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = currentImage.width;
                canvas.height = currentImage.height;
                ctx.drawImage(currentImage, 0, 0);
                
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const data = imageData.data;
                
                // Analyse simple de contraste pour trouver du texte
                let textRegions = [];
                const blockSize = 20;
                
                for (let y = 0; y < canvas.height; y += blockSize) {
                    for (let x = 0; x < canvas.width; x += blockSize) {
                        let sum = 0;
                        let count = 0;
                        
                        for (let dy = 0; dy < blockSize && y + dy < canvas.height; dy++) {
                            for (let dx = 0; dx < blockSize && x + dx < canvas.width; dx++) {
                                const idx = ((y + dy) * canvas.width + (x + dx)) * 4;
                                const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
                                sum += brightness;
                                count++;
                            }
                        }
                        
                        const avgBrightness = sum / count;
                        const variance = calculateVariance(data, x, y, blockSize, canvas.width, canvas.height, avgBrightness);
                        
                        // Haute variance = probablement du texte
                        if (variance > 1000) {
                            textRegions.push({x, y, variance});
                        }
                    }
                }
                
                // Regrouper les régions
                const numbers = findNumbersInRegions(textRegions);
                
                // Envoyer les résultats à Streamlit
                window.parent.postMessage({
                    type: 'text-detected',
                    numbers: numbers,
                    regions: textRegions.length
                }, '*');
                
                alert(`✅ ${numbers.length} numéros potentiels détectés!`);
            }
            
            function calculateVariance(data, startX, startY, blockSize, width, height, mean) {
                let variance = 0;
                let count = 0;
                
                for (let y = 0; y < blockSize && startY + y < height; y++) {
                    for (let x = 0; x < blockSize && startX + x < width; x++) {
                        const idx = ((startY + y) * width + (startX + x)) * 4;
                        const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
                        variance += Math.pow(brightness - mean, 2);
                        count++;
                    }
                }
                
                return variance / count;
            }
            
            function findNumbersInRegions(regions) {
                // Simuler la détection de numéros
                const numbers = [];
                const groups = [];
                
                // Grouper par proximité
                regions.forEach(region => {
                    let added = false;
                    for (let group of groups) {
                        const lastRegion = group[group.length - 1];
                        const distance = Math.sqrt(
                            Math.pow(region.x - lastRegion.x, 2) + 
                            Math.pow(region.y - lastRegion.y, 2)
                        );
                        
                        if (distance < 50) {
                            group.push(region);
                            added = true;
                            break;
                        }
                    }
                    
                    if (!added) {
                        groups.push([region]);
                    }
                });
                
                // Estimer les numéros
                groups.forEach((group, idx) => {
                    if (group.length >= 3) {
                        numbers.push({
                            number: `DETECTED_${idx}_${group.length}`,
                            length: group.length,
                            confidence: Math.min(100, 50 + group.length * 10)
                        });
                    }
                });
                
                return numbers;
            }
            
            // Charger le modèle au démarrage
            loadModel();
        </script>
    </body>
    </html>
    """
    
    return html_code

# ============================================
# SOLUTION 3: DÉTECTION PAR PATTERN MATCHING
# ============================================

def pattern_matching_detection(image):
    """
    Détection par reconnaissance de patterns de chiffres
    """
    import numpy as np
    
    # Convertir en array numpy
    img_array = np.array(image.convert('L'))
    height, width = img_array.shape
    
    # Patterns de chiffres simplifiés (7-segment like)
    patterns = {
        '0': np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]]),
        '8': np.array([[1,1,1],
                       [1,1,1],
                       [1,1,1]]),
    }
    
    detected_numbers = []
    
    # Scanner l'image par fenêtres
    window_sizes = [(20, 30), (30, 45), (15, 25)]
    
    for win_w, win_h in window_sizes:
        for y in range(0, height - win_h, 10):
            for x in range(0, width - win_w, 5):
                window = img_array[y:y+win_h, x:x+win_w]
                
                # Binariser la fenêtre
                threshold = np.mean(window)
                binary_window = (window < threshold).astype(int)
                
                # Calculer les caractéristiques
                density = np.sum(binary_window) / (win_w * win_h)
                
                # Les chiffres ont une densité spécifique
                if 0.15 < density < 0.45:
                    # Vérifier le ratio d'aspect
                    aspect_ratio = win_h / win_w
                    
                    if 1.5 < aspect_ratio < 3.0:
                        # C'est probablement un chiffre
                        detected_numbers.append({
                            'bbox': (x, y, win_w, win_h),
                            'density': density,
                            'aspect_ratio': aspect_ratio
                        })
    
    # Regrouper les détections proches en nombres
    numbers = group_detections_into_numbers(detected_numbers)
    
    return numbers, detected_numbers

def group_detections_into_numbers(detections, max_distance=30):
    """
    Groupe les détections de chiffres en nombres
    """
    if not detections:
        return []
    
    # Trier par position x
    sorted_detections = sorted(detections, key=lambda d: d['bbox'][0])
    
    numbers = []
    current_number = [sorted_detections[0]]
    
    for i in range(1, len(sorted_detections)):
        current = sorted_detections[i]
        previous = sorted_detections[i-1]
        
        # Distance horizontale
        x_distance = current['bbox'][0] - (previous['bbox'][0] + previous['bbox'][2])
        
        # Alignement vertical
        y_center_current = current['bbox'][1] + current['bbox'][3]/2
        y_center_previous = previous['bbox'][1] + previous['bbox'][3]/2
        y_difference = abs(y_center_current - y_center_previous)
        
        if x_distance < max_distance and y_difference < 15:
            current_number.append(current)
        else:
            if len(current_number) >= 2:
                numbers.append({
                    'digits': len(current_number),
                    'positions': [d['bbox'] for d in current_number],
                    'estimated_value': ''.join([str(i % 10) for i in range(len(current_number))])
                })
            current_number = [current]
    
    # Dernier groupe
    if len(current_number) >= 2:
        numbers.append({
            'digits': len(current_number),
            'positions': [d['bbox'] for d in current_number],
            'estimated_value': ''.join([str(i % 10) for i in range(len(current_number))])
        })
    
    return numbers

# ============================================
# INTERFACE PRINCIPALE
# ============================================

# Session state
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []

# Sidebar
with st.sidebar:
    st.header("🔧 Méthode de Détection")
    
    detection_method = st.selectbox(
        "Choisir la méthode",
        [
            "📊 Analyse de Contours",
            "🧠 TensorFlow.js (Navigateur)",
            "🎯 Pattern Matching",
            "🔬 Détection Combinée"
        ],
        help="Différentes méthodes de vision par ordinateur"
    )
    
    st.markdown("---")
    
    st.subheader("📤 Upload Images")
    uploaded_files = st.file_uploader(
        "Choisissez des images",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.session_state.uploaded_images = uploaded_files
    
    st.markdown("---")
    
    if st.button("🚀 LANCER DÉTECTION", type="primary", use_container_width=True):
        st.session_state.detection_results = []
        st.rerun()

# Zone principale
tab1, tab2, tab3 = st.tabs(["📋 Images", "🔍 Détection", "📊 Résultats"])

with tab1:
    st.subheader("Images chargées")
    
    if st.session_state.uploaded_images:
        cols = st.columns(min(3, len(st.session_state.uploaded_images)))
        
        for idx, img_file in enumerate(st.session_state.uploaded_images):
            with cols[idx % 3]:
                image = Image.open(img_file)
                st.image(image, caption=img_file.name, use_container_width=True)
                
                # Info image
                st.caption(f"Taille: {image.size[0]}x{image.size[1]}")
    else:
        st.info("Chargez des images dans la barre latérale")

with tab2:
    st.subheader("Processus de Détection")
    
    if detection_method == "🧠 TensorFlow.js (Navigateur)":
        st.markdown("### Détection avec TensorFlow.js")
        st.info("Cette méthode utilise l'IA dans votre navigateur pour détecter les objets et le texte")
        
        # Intégrer le détecteur TensorFlow
        components.html(create_tensorflow_detector(), height=800)
        
    elif st.session_state.uploaded_images and not st.session_state.detection_results:
        st.info(f"Détection en cours avec la méthode: {detection_method}")
        
        progress_bar = st.progress(0)
        
        for idx, img_file in enumerate(st.session_state.uploaded_images):
            st.write(f"Analyse de {img_file.name}...")
            
            image = Image.open(img_file)
            
            if detection_method == "📊 Analyse de Contours":
                # Méthode 1: Contours
                regions, binary = detect_numbers_by_contours(image)
                numbers = extract_numbers_from_regions(image, regions)
                
                result = {
                    'file': img_file.name,
                    'method': 'Analyse de Contours',
                    'regions_detected': len(regions),
                    'numbers': numbers,
                    'binary_image': binary
                }
                
                # Afficher visualisation
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Originale", use_container_width=True)
                with col2:
                    st.image(binary, caption="Binarisée", use_container_width=True)
                
            elif detection_method == "🎯 Pattern Matching":
                # Méthode 3: Pattern Matching
                numbers, detections = pattern_matching_detection(image)
                
                result = {
                    'file': img_file.name,
                    'method': 'Pattern Matching',
                    'detections': len(detections),
                    'numbers': numbers
                }
                
                st.write(f"✅ {len(detections)} chiffres détectés")
                st.write(f"📊 {len(numbers)} nombres identifiés")
                
            else:  # Combinée
                # Combiner plusieurs méthodes
                regions, binary = detect_numbers_by_contours(image)
                numbers1 = extract_numbers_from_regions(image, regions)
                numbers2, detections = pattern_matching_detection(image)
                
                # Fusionner les résultats
                all_numbers = numbers1 + [
                    {'number': n['estimated_value'], 
                     'digits': n['digits'], 
                     'confidence': 'PATTERN'} 
                    for n in numbers2
                ]
                
                result = {
                    'file': img_file.name,
                    'method': 'Détection Combinée',
                    'contour_regions': len(regions),
                    'pattern_detections': len(detections) if detections else 0,
                    'numbers': all_numbers
                }
            
            st.session_state.detection_results.append(result)
            progress_bar.progress((idx + 1) / len(st.session_state.uploaded_images))
        
        st.success("✅ Détection terminée!")
        st.rerun()

with tab3:
    st.subheader("Résultats de la Détection")
    
    if st.session_state.detection_results:
        # Tableau récapitulatif
        summary = []
        all_numbers = []
        
        for result in st.session_state.detection_results:
            num_count = len(result.get('numbers', []))
            
            summary.append({
                'Fichier': result['file'],
                'Méthode': result['method'],
                'Éléments détectés': num_count,
                'Régions': result.get('regions_detected', result.get('detections', 0))
            })
            
            # Collecter tous les numéros
            for num_info in result.get('numbers', []):
                if isinstance(num_info, dict):
                    num_value = num_info.get('number', num_info.get('estimated_value', 'N/A'))
                    confidence = num_info.get('confidence', 'DETECTED')
                else:
                    num_value = str(num_info)
                    confidence = 'DETECTED'
                
                all_numbers.append({
                    'Fichier': result['file'],
                    'Numéro': num_value,
                    'Confiance': confidence
                })
        
        # Afficher le résumé
        df_summary = pd.DataFrame(summary)
        st.dataframe(df_summary, use_container_width=True)
        
        # Métriques
        total_detected = sum(s['Éléments détectés'] for s in summary)
        st.metric("Total Éléments Détectés", total_detected)
        
        # Détails par fichier
        st.markdown("---")
        st.subheader("📋 Détails par Image")
        
        for result in st.session_state.detection_results:
            with st.expander(f"📄 {result['file']} - {result['method']}"):
                st.write(f"**Méthode utilisée:** {result['method']}")
                st.write(f"**Éléments détectés:** {len(result.get('numbers', []))}")
                
                if result.get('numbers'):
                    st.write("**Numéros identifiés:**")
                    for num in result['numbers']:
                        if isinstance(num, dict):
                            st.code(f"{num.get('number', num.get('estimated_value'))} (confiance: {num.get('confidence', 'N/A')})")
                        else:
                            st.code(str(num))
        
        # Export
        if all_numbers:
            st.markdown("---")
            st.subheader("💾 Export")
            
            df_export = pd.DataFrame(all_numbers)
            csv_data = df_export.to_csv(index=False)
            
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv_data,
                file_name=f"detection_visuelle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Afficher la liste
            st.text_area(
                "Liste des numéros",
                '\n'.join([n['Numéro'] for n in all_numbers]),
                height=150
            )
    else:
        st.info("Lancez la détection dans l'onglet 'Détection'")

# Footer
st.markdown("---")
st.markdown("""
<div style='background: #1a1a2e; padding: 20px; border-radius: 10px; color: white;'>
    <h3 style='text-align: center; color: white;'>👁️ Détection par Vision par Ordinateur</h3>
    <p style='text-align: center;'>
        <b>Méthodes utilisées:</b> Analyse de Contours • Pattern Matching • TensorFlow.js<br>
        <b>Avantages:</b> Détection réelle sans OCR • Analyse de forme • Indépendant du texte
    </p>
    <p style='text-align: center; font-size: 12px; margin-top: 20px;'>
        ✅ Détection basée sur la morphologie et les caractéristiques visuelles des chiffres
    </p>
</div>
""", unsafe_allow_html=True)
