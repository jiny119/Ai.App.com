import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import base64
import os

# Load Model
@st.cache_resource()
def load_model():
    model_name = "urduhack/urdu-gpt2"  # Replace with an appropriate Urdu model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Generate Story Function
def generate_story(title, max_length=4000):
    input_text = f"{title} - "
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_length=max_length, 
            num_return_sequences=1, 
            temperature=0.7
        )

    story = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return story

# Convert text to speech
def text_to_speech(text, lang="ur"):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("story.mp3")
    return "story.mp3"

# Background Music (Add your music file URL)
background_music_url = "https://example.com/background.mp3"

# Streamlit UI
st.title("🎙 Urdu Story Generator with Voiceover & Music")
title = st.text_input("Enter Story Title:", "جادوئی کہانی")

if st.button("Generate Story"):
    with st.spinner("Generating..."):
        story = generate_story(title)
        audio_file = text_to_speech(story)
        
        st.subheader("📖 Generated Story")
        st.write(story)

        st.subheader("🎵 Background Music")
        st.audio(background_music_url, format="audio/mp3")

        st.subheader("🔊 Voiceover")
        st.audio(audio_file, format="audio/mp3")

        with open(audio_file, "rb") as file:
            btn = st.download_button(label="Download Voiceover", data=file, file_name="urdu_story.mp3", mime="audio/mp3")
