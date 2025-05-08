import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename="interview.wav", duration=10, fs=16000):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  
    write(filename, fs, audio)
    print("Done recording.")

record_audio()