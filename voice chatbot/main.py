import streamlit as st
import os
import warnings
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
import whisper

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

# Load the Whisper model
model = whisper.load_model("base", device="cpu")

audio_bytes = audio_recorder()

if audio_bytes:
    temp_folder = 'temp'
    os.makedirs(temp_folder, exist_ok=True) 

    audio_path = os.path.join(temp_folder, "audio.wav")
    
    # Save audio_bytes to the specified path
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)
    
    # Transcribe the audio file with Whisper
    result = model.transcribe(audio_path, language="en")
    st.write(result['text'])
    
    os.remove(audio_path)
