import streamlit as st
from src.cleaner import clean_text
from src.processor import process_text
from src.exporter import save_to_csv

st.title("🧠 Text Analyzer App")

text = st.text_area("Enter your text")

if st.button("Analyze"):
    cleaned = clean_text(text)
    data, entities = process_text(cleaned)

    st.subheader("📊 Tokens")
    st.write(data)

    st.subheader("🏷️ Named Entities")
    st.write(entities)

    save_to_csv(data)

    st.success("Results saved to CSV!")