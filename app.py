import streamlit as st
from translation import for_translation
import pandas
import numpy


st.title("Text to Text Translator")

tab1, = st.tabs(["Spanish"])

with tab1:
    st.text_input("Enter the sentence", key="sentence")

    st.subheader("Translated text:")
    if st.session_state.sentence:
        translated = for_translation(sentence=st.session_state.sentence)
        st.write(translated)
