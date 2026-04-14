import streamlit as st
import pandas as pd
import re
import io
from PIL import Image
import pytesseract

st.set_page_config(page_title="Extracteur Numéros", page_icon="📸")
st.title("📸 Extraire les numéros des photos")

# Upload des photos
fichiers = st.file_uploader(
    "Choisissez des photos", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if fichiers:
    resultats = []
    
    for fichier in fichiers:
        image = Image.open(fichier)
        st.image(image, caption=fichier.name, width=400)
        
        with st.spinner(f"🔍 Analyse de {fichier.name}..."):
            # OCR avec Tesseract
            try:
                texte = pytesseract.image_to_string(image)
                
                # Extraire tous les nombres
                numeros = re.findall(r'\d+', texte)
                
                if numeros:
                    st.success(f"✅ {len(numeros)} numéro(s) trouvé(s) : {', '.join(numeros)}")
                else:
                    st.warning("⚠️ Aucun numéro trouvé")
                
                resultats.append({
                    "Fichier": fichier.name,
                    "Numéros extraits": ", ".join(numeros) if numeros else "Aucun",
                    "Quantité": len(numeros)
                })
                
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")
    
    # Afficher les résultats
    if resultats:
        st.divider()
        st.subheader("📊 Tableau des résultats")
        
        df = pd.DataFrame(resultats)
        st.dataframe(df, use_container_width=True)
        
        # Bouton Export Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Numéros')
        
        st.download_button(
            label="📥 Télécharger le fichier Excel",
            data=output.getvalue(),
            file_name="numeros_extraits.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
else:
    st.info("👆 Téléchargez une ou plusieurs photos pour commencer")
