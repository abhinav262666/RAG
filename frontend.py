import streamlit as st
import requests
import base64
from io import BytesIO

# Sarvam API details for Text-to-Speech
SARVAM_TTS_API_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_API_KEY = "4d4da973-afd1-45ec-8999-6d8f077c98e1"  # Replace with your valid API key

# Function to handle Text-to-Speech conversion via Sarvam API
def convert_text_to_speech(text_output, language_code="hi-IN", speaker="meera", pitch=None, pace=None, loudness=None, speech_sample_rate=22050, enable_preprocessing=False, model="bulbul:v1"):
    try:
        # Prepare the JSON payload
        payload = {
            "inputs": [text_output],
            "target_language_code": language_code,
            "speaker": speaker,
            "pitch": pitch,
            "pace": pace,
            "loudness": loudness,
            "speech_sample_rate": speech_sample_rate,
            "enable_preprocessing": enable_preprocessing,
            "model": model
        }
        
        # Headers with API key
        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": SARVAM_API_KEY
        }
        
        # Send POST request to Sarvam Text-to-Speech API
        response = requests.post(SARVAM_TTS_API_URL, headers=headers, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            audio_base64 = response.json()["audios"][0]  # Get the base64 encoded audio
            return audio_base64
        else:
            return f"Error: {response.status_code} - {response.json().get('detail', 'Something went wrong')}"
    
    except Exception as e:
        return f"Error connecting to Sarvam API: {e}"

# Function to convert base64 audio to playable format
def play_audio(base64_audio):
    # Decode base64 string into bytes
    audio_bytes = base64.b64decode(base64_audio)
    
    # Play the audio using Streamlit
    st.audio(audio_bytes, format="audio/wav")

# Function to handle weather queries
def get_weather(query):
    try:
        response = requests.post(f"http://localhost:8000/generate-weather/", json={"query": query})
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code} - {response.json().get('detail', 'Something went wrong')}"
    except Exception as e:
        return f"Error connecting to the server: {e}"

# Function to handle Study Chapter 11 queries
def get_study_response(query):
    try:
        response = requests.post(f"http://localhost:8000/generate-response/", json={"query": query})
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code} - {response.json().get('detail', 'Something went wrong')}"
    except Exception as e:
        return f"Error connecting to the server: {e}"

# Streamlit App UI
st.title("FastAPI-Streamlit with Text-to-Speech")

# Sidebar with options
option = st.sidebar.selectbox("Choose an option:", ["Weather", "Study Chapter 11: Sound"])

if option == "Weather":
    st.header("Get Weather Information")
    
    # Text input for weather query
    user_query = st.text_input("Ask about the weather (e.g., 'What is the weather in New York?')", "")
    
    if st.button("Get Weather"):
        if user_query:
            # Call the FastAPI weather endpoint
            result = get_weather(user_query)
            st.text_area("Weather Response", value=result, height=200)
            
            # Convert the response text to speech
            tts_audio_base64 = convert_text_to_speech(result, language_code="en-IN", speaker="meera")
            
            # Play the audio if successfully converted
            if isinstance(tts_audio_base64, str) and tts_audio_base64.startswith("Error"):
                st.error(tts_audio_base64)  # Show error if conversion fails
            else:
                st.write("Playing the response as speech:")
                play_audio(tts_audio_base64)
        else:
            st.warning("Please enter a query.")

elif option == "Study Chapter 11: Sound":
    st.header("Study Chapter 11: Sound with Ollama")
    
    # Text input for study query
    user_query = st.text_input("Ask about Sound (e.g., 'Explain how sound waves propagate.')", "")
    
    if st.button("Get Answer"):
        if user_query:
            # Call the FastAPI Ollama study endpoint
            result = get_study_response(user_query)
            st.text_area("Ollama's Response", value=result, height=200)
            
            # Convert the response text to speech
            tts_audio_base64 = convert_text_to_speech(result, language_code="en-IN", speaker="meera")
            
            # Play the audio if successfully converted
            if isinstance(tts_audio_base64, str) and tts_audio_base64.startswith("Error"):
                st.error(tts_audio_base64)  # Show error if conversion fails
            else:
                st.write("Playing the response as speech:")
                play_audio(tts_audio_base64)
        else:
            st.warning("Please enter a query.")
