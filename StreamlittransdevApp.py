""")
with col2:
st.markdown("""
**Numeros qui seront detectes:**
- ✅ **10406871** (Heures)
- ✅ **623743** (Kilometrage)
""")

st.success("🎯 L'application detecte automatiquement ces valeurs!")

with tab2:
st.header("Analyse OCR en cours")

if st.session_state.processing:
st.info("🔄 Traitement des photos...")

progress_bar = st.progress(0)
status_text = st.empty()

for idx, img_file in enumerate(st.session_state.uploaded_images):
status_text.text(f"📸 Analyse de {img_file.name} ({idx+1}/{len(st.session_state.uploaded_images)})")

image = Image.open(img_file)

col1, col2 = st.columns([1, 2])

with col1:
    st.image(image, caption=f"Original: {img_file.name}", use_container_width=True)

with col2:
    with st.spinner("🔍 Tesseract OCR en cours..."):
        # Extraction des numeros
        numbers, full_text = extract_numbers_from_image(image)
        
        # Filtrer par longueur
        numbers = [n for n in numbers if len(n) >= min_digits]
        
        # Classifier les numeros
        classified = classify_numbers(numbers, full_text)
        
        result = {
            'filename': img_file.name,
            'heures': classified['heures'],
            'kilometrages': classified['kilometrages'],
            'autres': classified['autres'],
            'all_numbers': numbers,
            'full_text': full_text,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Afficher les resultats
        if classified['heures']:
            st.success(f"✅ Heures: {', '.join(classified['heures'])}")
        if classified['kilometrages']:
            st.success(f"✅ Kilometrage: {', '.join(classified['kilometrages'])}")
        
        if not numbers:
            st.warning("⚠️ Aucun numero detecte")
        
        st.session_state.results.append(result)

progress_bar.progress((idx + 1) / len(st.session_state.uploaded_images))

status_text.text("✅ Analyse terminee!")
st.session_state.processing = False

st.success("🎉 Toutes les photos ont ete analysees!")
st.rerun()

else:
if st.session_state.uploaded_images:
st.info("👆 Cliquez sur 'ANALYSER' dans la barre laterale")

st.markdown("### 📸 Apercu des photos")
cols = st.columns(min(3, len(st.session_state.uploaded_images)))

for idx, img_file in enumerate(st.session_state.uploaded_images):
    with cols[idx % 3]:
        image = Image.open(img_file)
        st.image(image, caption=img_file.name, width=200)
else:
st.info("📤 Uploadez des photos")

with tab3:
st.header("Resultats de l'extraction")

if st.session_state.results:
# Resume
st.subheader("📊 Resume des extractions")

summary_data = []
all_heures = []
all_km = []

for r in st.session_state.results:
summary_data.append({
    'Fichier': r['filename'],
    'Heures': len(r['heures']),
    'Kilometrages': len(r['kilometrages']),
    'Autres': len(r['autres']),
    'Date': r['timestamp']
})
all_heures.extend(r['heures'])
all_km.extend(r['kilometrages'])

df_summary = pd.DataFrame(summary_data)
st.dataframe(df_summary, use_container_width=True)

# Metriques
col1, col2, col3, col4 = st.columns(4)

with col1:
st.metric("📁 Fichiers", len(st.session_state.results))
with col2:
st.metric("⏱️ Heures", len(all_heures))
with col3:
st.metric("🛣️ Kilometrages", len(all_km))
with col4:
st.metric("🔢 Total numeros", len(all_heures) + len(all_km))

st.markdown("---")

# Details par fichier
st.subheader("📋 Details par photo")

for idx, result in enumerate(st.session_state.results):
with st.expander(f"📄 {result['filename']}"):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        img_file = st.session_state.uploaded_images[idx] if idx < len(st.session_state.uploaded_images) else None
        if img_file:
            image = Image.open(img_file)
            st.image(image, width=250)
    
    with col2:
        st.write(f"**Date d'analyse:** {result['timestamp']}")
        
        if result['heures']:
            st.success("**⏱️ Nombres d'heures:**")
            for num in result['heures']:
                st.code(num, language='text')
        
        if result['kilometrages']:
            st.success("**🛣️ Kilometrages:**")
            for num in result['kilometrages']:
                st.code(num, language='text')
        
        if result['autres']:
            st.info("**📌 Autres numeros:**")
            for num in result['autres']:
                st.code(num, language='text')
        
        if result['full_text']:
            with st.expander("📝 Voir le texte complet extrait"):
                st.text(result['full_text'])

st.markdown("---")

# Export
st.subheader("💾 Export des resultats")

# Preparer les donnees pour export
export_data = []
for r in st.session_state.results:
for num in r['heures']:
    export_data.append({
        'Fichier': r['filename'],
        'Type': 'Heures',
        'Valeur': num,
        'Date': r['timestamp']
    })
for num in r['kilometrages']:
    export_data.append({
        'Fichier': r['filename'],
        'Type': 'Kilometrage',
        'Valeur': num,
        'Date': r['timestamp']
    })
for num in r['autres']:
    export_data.append({
        'Fichier': r['filename'],
        'Type': 'Autre',
        'Valeur': num,
        'Date': r['timestamp']
    })

if export_data:
df_export = pd.DataFrame(export_data)

# CSV
csv_data = df_export.to_csv(index=False, encoding='utf-8-sig')
st.download_button(
    label="📥 TELECHARGER CSV",
    data=csv_data,
    file_name=f"compteur_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
    use_container_width=True
)

# Afficher resume texte
st.markdown("---")
st.subheader("📋 Resume texte")

resume_text = ""
for r in st.session_state.results:
    resume_text += f"\n{fichier}: {r['filename']}\n"
    if r['heures']:
        resume_text += f"  Heures: {', '.join(r['heures'])}\n"
    if r['kilometrages']:
        resume_text += f"  Kilometrage: {', '.join(r['kilometrages'])}\n"

st.text_area("Resume (copiable)", resume_text, height=200)

else:
st.info("🔍 Aucun resultat disponible. Lancez une analyse dans l'onglet 'Analyse'.")

# Footer
st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 10px; color: white; text-align: center;'>
<h3 style='color: white;'>🔢 Extracteur de Numeros - Compteurs</h3>
<p>Detection automatique des heures et kilometrages</p>
<p style='font-size: 12px;'>Tesseract OCR | Compatible Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)
