import streamlit as st
from transformers import pipeline
import os

# Hugging Face API anahtarını buraya girin
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_senin_token_kodun"

st.set_page_config(page_title="MediGuide", page_icon="🧬")

st.title("🩺 MediGuide - Sağlık Asistanı AI")
st.write("📌 Sağlıkla ilgili sorularınızı yazın, yapay zeka yardımcı olsun.")

@st.cache_resource
def get_model():
    return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", 
                    tokenizer="mistralai/Mistral-7B-Instruct-v0.1", 
                    max_length=256, 
                    temperature=0.7)

qa_pipeline = get_model()

soru = st.text_input("🧠 Sorunuzu yazın:", placeholder="Örn: Baş ağrısına ne iyi gelir?")

if soru:
    with st.spinner("Cevap hazırlanıyor..."):
        yanit = qa_pipeline(soru)[0]['generated_text']
        st.success(f"💬 Cevap: {yanit}")