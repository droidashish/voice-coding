import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from pynput.keyboard import Controller

model = WhisperModel("base.en", device="cpu")  # small + fast
keyboard = Controller()

SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # 0.25 sec

audio_buffer = np.array([], dtype=np.float32)

def callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = np.append(audio_buffer, indata[:, 0])

    if len(audio_buffer) > SAMPLE_RATE * 3:   # every 3 sec
        process_audio()

def process_audio():
    global audio_buffer
    global model

    audio_chunk = audio_buffer.copy()
    audio_buffer = np.array([], dtype=np.float32)

    segments, _ = model.transcribe(audio_chunk, beam_size=1)
    
    for seg in segments:
        text = seg.text.strip()
        if text:
            for char in text + " ":
                keyboard.press(char)
                keyboard.release(char)

# Start listening
with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE):
    print("🎤 Listening... Speak anything...")
    while True:
        sd.sleep(1000)
