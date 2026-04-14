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
        
        prompt = """
        Analyse cette image et extrait TOUS les numeros visibles.
        
        INSTRUCTIONS:
        1. Trouve TOUS les nombres dans l'image (comme 10406871, 823743, etc.)
        2. Extrait egalement tout texte visible
        
        Reponds UNIQUEMENT avec un JSON valide selon ce format exact:
        {
            "success": true,
            "numbers": ["10406871", "823743"],
            "full_text": "texte complet ici",
            "confidence": "high"
        }
        
        IMPORTANT: Ne mets PAS de texte avant ou apres le JSON. Juste le JSON pur.
        """
        
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
            
            st.markdown("""
            **Image type (comme TESTO01.jpeg):**
