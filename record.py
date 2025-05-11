import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

def record_audio(filename="interviewaudio.wav", fs=16000, channels=1):
    print("Recording…  Type ‘stop’ + Enter to finish.")
    frames = []

    def callback(indata, _frames, _time, status):
        if status:
            print(f"⚠️  {status}")
        frames.append(indata.copy())

    # Open the stream in the background
    with sd.InputStream(samplerate=fs, channels=channels, dtype='int16', callback=callback):
        # Block here until user types ‘stop’
        while True:
            cmd = input()
            if cmd.strip().lower() == "stop":
                break

    # Concatenate all of the recorded chunks and write
    audio = np.concatenate(frames, axis=0)
    write(filename, fs, audio)
    print(f"Done recording. Saved to {filename}")

#def record_audio(filename="interviewaudio.wav", duration=10, fs=16000):
#    print("Recording...")
#    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
#    sd.wait()  
#    write(filename, fs, audio)
#    print("Done recording.")