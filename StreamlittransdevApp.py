import streamlit as st
import pandas as pd
from PIL import Image
import io
import re
from datetime import datetime
import google.generativeai as genai
import json
import time

# Configuration de la page
st.set_page_config(
    page_title="Extracteur de Numeros AI",
    page_icon="123",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration Gemini AI
GEMINI_API_KEY = "AIzaSyCYBqDiM7-YMu2MoecgYpg30U3ONdYDA8A"

def initialize_gemini():
    """Initialise le modele Gemini avec la cle API"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config
        )
        
        return model, True, "Gemini initialise avec succes"
    except Exception as e:
        return None, False, f"Erreur d'initialisation: {str(e)}"

def extract_numbers_with_gemini(image, model):
    """Utilise Gemini pour extraire les numeros d'une image"""
    try:
        img_byte_arr = io.BytesIO()
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(img_byte_arr, format='JPEG', quality=95)
        img_bytes = img_byte_arr.getvalue()
        
        prompt = "Analyse cette image et extrait TOUS les numeros visibles. Reponds UNIQUEMENT avec un JSON valide selon ce format exact: {\"success\": true, \"numbers\": [\"10406871\", \"823743\"], \"full_text\": \"texte complet ici\", \"confidence\": \"high\"}"
        
        contents = [
            prompt,
            {"mime_type": "image/jpeg", "data": img_bytes}
        ]
        
        response = model.generate_content(contents)
        response_text = response.text.strip()
        
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        elif response_text.startswith('```'):
            response_text = response_text[3:]
        
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        return result
        
    except json.JSONDecodeError:
        try:
            simple_prompt = "Liste uniquement les nombres visibles dans cette image, separes par des virgules."
            contents_simple = [simple_prompt, {"mime_type": "image/jpeg", "data": img_bytes}]
            response_simple = model.generate_content(contents_simple)
            
            numbers = re.findall(r'\b\d{2,}\b', response_simple.text)
            
            return {
                "success": True,
                "numbers": list(set(numbers)),
                "full_text": response_simple.text,
                "confidence": "medium"
            }
        except:
            return {
                "success": False,
                "error": "Erreur de parsing",
                "numbers": [],
                "full_text": "",
                "confidence": "error"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "numbers": [],
            "full_text": "",
            "confidence": "error"
        }

def extract_numbers_regex(text):
    """Extraction complementaire par regex"""
    if not text:
        return []
    
    patterns = [
        r'\b\d{5,}\b',
        r'\b\d{2,4}\b',
        r'\b\d+[\s-]?\d+\b',
    ]
    
    all_numbers = set()
    for pattern in patterns:
        numbers = re.findall(pattern, text)
        for num in numbers:
            clean_num = re.sub(r'[\s-]', '', num)
            if clean_num.isdigit() and len(clean_num) >= 2:
                all_numbers.add(clean_num)
    
    return sorted(list(all_numbers), key=len, reverse=True)

def preprocess_image(image):
    """Pretraite l'image pour ameliorer la detection"""
    max_size = 2000
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image

def create_csv_export(results):
    """Cree un fichier CSV avec les resultats"""
    data = []
    for result in results:
        for number in result.get('numbers', []):
            data.append({
                'Fichier': result['filename'],
                'Numero': number,
                'Confiance': result.get('confidence', 'N/A'),
                'Date': result['timestamp']
            })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False, encoding='utf-8-sig')

def main():
    """Fonction principale de l'application"""
    
    st.title("Extracteur de Numeros avec Google Gemini AI")
    st.markdown("---")
    
    # Initialiser Gemini
    if 'gemini_model' not in st.session_state:
        model, success, message = initialize_gemini()
        st.session_state.gemini_model = model
        st.session_state.gemini_ready = success
        st.session_state.gemini_message = message
    
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
            accept_multiple_files=True,
            help="Selectionnez une ou plusieurs photos"
        )
        
        if uploaded_files:
            st.session_state.uploaded_images = uploaded_files
            st.success(f"{len(uploaded_files)} image(s) chargee(s)")
        
        st.markdown("---")
        
        st.header("Statut AI")
        if st.session_state.gemini_ready:
            st.success(st.session_state.gemini_message)
            st.info("1500 requetes/jour gratuites")
        else:
            st.error(st.session_state.gemini_message)
        
        st.markdown("---")
        
        st.header("Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("ANALYSER", type="primary", use_container_width=True):
                if not st.session_state.uploaded_images:
                    st.warning("Chargez d'abord des images")
                elif not st.session_state.gemini_ready:
                    st.error("Gemini n'est pas initialise")
                else:
                    st.session_state.processing = True
                    st.session_state.results = []
                    st.rerun()
        
        with col2:
            if st.button("REINITIALISER", use_container_width=True):
                st.session_state.uploaded_images = []
                st.session_state.results = []
                st.session_state.processing = False
                st.rerun()
        
        st.markdown("---")
        
        st.header("Statistiques")
        st.metric("Images chargees", len(st.session_state.uploaded_images))
        st.metric("Resultats", len(st.session_state.results))
        
        if st.session_state.results:
            total_numbers = sum(len(r.get('numbers', [])) for r in st.session_state.results)
            st.metric("Total numeros", total_numbers)
    
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
                    
                    st.caption(f"Taille: {image.size[0]}x{image.size[1]}")
                    
                    if st.button(f"Supprimer", key=f"del_{idx}"):
                        st.session_state.uploaded_images.pop(idx)
                        st.rerun()
        else:
            st.info("Chargez des images dans la barre laterale")
            
            st.markdown("---")
            st.markdown("### Exemple de detection")
            
            st.write("Image type (comme TESTO01.jpeg):")
            st.code("HENGSTLER\n10406871\n823743")
            st.write("Numeros qui seront detectes: 10406871, 823743")
    
    # Tab 2: Analyse
    with tab2:
        st.header("Analyse avec Gemini AI")
        
        if st.session_state.processing:
            st.info("Traitement en cours...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, img_file in enumerate(st.session_state.uploaded_images):
                status_text.text(f"Analyse de {img_file.name} ({idx+1}/{len(st.session_state.uploaded_images)})")
                
                image = Image.open(img_file)
                image = preprocess_image(image)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(image, caption=f"Analyse: {img_file.name}", width=200)
                
                with col2:
                    with st.spinner("Gemini analyse..."):
                        result = extract_numbers_with_gemini(image, st.session_state.gemini_model)
                        result['filename'] = img_file.name
                        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        if result.get('success'):
                            st.success(f"{len(result.get('numbers', []))} numeros trouves")
                            st.write(f"Confiance: {result.get('confidence', 'N/A')}")
                        else:
                            st.error(f"Erreur: {result.get('error', 'Inconnue')}")
                        
                        if result.get('numbers'):
                            st.write("Numeros detectes:")
                            for num in result['numbers']:
                                st.code(num)
                        
                        st.session_state.results.append(result)
                
                progress_bar.progress((idx + 1) / len(st.session_state.uploaded_images))
                time.sleep(0.5)
            
            status_text.text("Analyse terminee!")
            st.session_state.processing = False
            st.rerun()
        
        else:
            if st.session_state.uploaded_images:
                st.info("Cliquez sur 'ANALYSER' dans la barre laterale pour commencer")
                
                st.markdown("### Apercu des images a analyser")
                cols = st.columns(min(3, len(st.session_state.uploaded_images)))
                
                for idx, img_file in enumerate(st.session_state.uploaded_images):
                    with cols[idx % 3]:
                        image = Image.open(img_file)
                        st.image(image, caption=img_file.name, width=150)
            else:
                st.info("Chargez d'abord des images dans la barre laterale")
    
    # Tab 3: Resultats
    with tab3:
        st.header("Resultats de l'extraction")
        
        if st.session_state.results:
            st.subheader("Recapitulatif")
            
            summary_data = []
            all_numbers = []
            
            for r in st.session_state.results:
                numbers_count = len(r.get('numbers', []))
                summary_data.append({
                    'Fichier': r['filename'],
                    'Numeros': numbers_count,
                    'Confiance': r.get('confidence', 'N/A'),
                    'Date': r['timestamp']
                })
                
                all_numbers.extend(r.get('numbers', []))
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fichiers", len(st.session_state.results))
            with col2:
                st.metric("Total Numeros", len(all_numbers))
            with col3:
                unique_numbers = len(set(all_numbers))
                st.metric("Numeros uniques", unique_numbers)
            
            st.markdown("---")
            
            st.subheader("Details par fichier")
            
            for idx, result in enumerate(st.session_state.results):
                with st.expander(f"{result['filename']} - {len(result.get('numbers', []))} numeros"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        img_file = st.session_state.uploaded_images[idx] if idx < len(st.session_state.uploaded_images) else None
                        if img_file:
                            image = Image.open(img_file)
                            st.image(image, width=200)
                    
                    with col2:
                        st.write(f"Confiance: {result.get('confidence', 'N/A')}")
                        st.write(f"Date d'analyse: {result['timestamp']}")
                        
                        if result.get('numbers'):
                            st.write("Numeros detectes:")
                            for num in result['numbers']:
                                st.code(num)
                        else:
                            st.warning("Aucun numero detecte")
                        
                        if result.get('full_text'):
                            st.text_area(
                                "Texte complet extrait",
                                result['full_text'][:500],
                                height=100,
                                key=f"text_{idx}"
                            )
            
            st.markdown("---")
            
            st.subheader("Export des resultats")
            
            if st.button("Telecharger CSV"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_data = create_csv_export(st.session_state.results)
                
                st.download_button(
                    label="Cliquer pour telecharger",
                    data=csv_data,
                    file_name=f"numeros_{timestamp}.csv",
                    mime="text/csv"
                )
            
            if all_numbers:
                st.markdown("---")
                st.subheader("Copier tous les numeros")
                
                numbers_text = '\n'.join(all_numbers)
                st.text_area(
                    "Tous les numeros (Ctrl+C pour copier)",
                    numbers_text,
                    height=150
                )
        
        else:
            st.info("Aucun resultat disponible. Lancez une analyse dans l'onglet 'Analyse'.")
            
            st.markdown("---")
            st.markdown("### Exemple de resultat attendu")
            
            example_data = pd.DataFrame([
                {"Fichier": "TESTO01.jpeg", "Numero": "10406871", "Confiance": "high"},
                {"Fichier": "TESTO01.jpeg", "Numero": "823743", "Confiance": "high"}
            ])
            
            st.dataframe(example_data, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;'>
        <h3 style='color: white;'>Extracteur de Numeros avec Google Gemini AI</h3>
        <p>Version 2.0 | API Gemini integree | 1500 requetes/jour gratuites</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
