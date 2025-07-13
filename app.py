import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="MediGuide", page_icon="🧬")

st.title("🩺 MediGuide - Sağlık Asistanı AI")
st.write("📌 Sağlıkla ilgili sorularınızı yazın, yapay zeka yardımcı olsun.")

@st.cache_resource
def get_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

qa_pipeline = get_model()

soru = st.text_input("🧠 Sorunuzu yazın:", placeholder="Örn: Karın ağrısına ne iyi gelir?")

if soru:
    with st.spinner("Cevap hazırlanıyor..."):
        yanit = qa_pipeline(soru, max_length=256)[0]['generated_text']
        st.success(f"💬 Cevap: {yanit}")
