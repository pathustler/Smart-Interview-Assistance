import sounddevice as sd
import soundfile as sf
import numpy as np
from pydub import AudioSegment, silence  # trim recorded audio

def record_audio(filename="interviewaudio.wav", fs=16000, channels=1):
    print("Recording... Type 'stop' + Enter to finish.")
    frames = []

    def callback(indata, _frames, _time, status):
        if status:
            print(f"⚠️ {status}")
        frames.append(indata.copy())

    with sd.InputStream(samplerate=fs, channels=channels, dtype='int16', callback=callback):
        while True:
            if input().strip().lower() == "stop":
                break

    # Save raw audio
    audio = np.concatenate(frames, axis=0)
    sf.write(filename, audio, fs, subtype='PCM_16')
    print(f"Done recording. Saved to {filename}")

    # Trim silence at start and end
    try:
        audio_seg = AudioSegment.from_wav(filename)
        # detect nonsilent (ms)
        nonsilence = silence.detect_nonsilent(
            audio_seg,
            min_silence_len=300,
            silence_thresh=audio_seg.dBFS - 16
        )
        if nonsilence:
            start_trim, end_trim = nonsilence[0][0], nonsilence[-1][1]
            trimmed = audio_seg[start_trim:end_trim]
            trimmed.export(filename, format="wav")
            print(f"Trimmed silence: start={start_trim}ms, end={end_trim}ms")
    except Exception as e:
        print(f"⚠️ Could not trim silence: {e}")