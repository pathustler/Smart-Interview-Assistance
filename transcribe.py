import warnings
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2' 
warnings.filterwarnings(
    'ignore',
    category=DeprecationWarning,
    module='tf_keras.src.losses'
)
from transformers import logging as tfm_logging
tfm_logging.set_verbosity_error()

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

import speech_recognition as sr
import language_tool_python
import textstat
from transformers import pipeline
import re
import numpy as np 
import myprosody as mysp
import wave

from record import record_audio
from pydub import AudioSegment, silence


# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

PROSODY_SCRIPT_PATH = "prosody_analysis.praat"

def get_audio_duration(file_path):
    with wave.open(file_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    return frames / rate

# === Helper Functions ===
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        print("Transcribing audio...")
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print("Transcription complete.")
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError:
        return "Error with the Google Speech Recognition service."

def grammar_feedback(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)

    # Rules to ignore: punctuation, capitalization, etc.
    excluded_rules = {
        "UPPERCASE_SENTENCE_START",
        "PUNCTUATION_PARAGRAPH_END",
        "PUNCTUATION",
        "COMMA_PARENTHESIS_WHITESPACE",
        "EN_QUOTES",
        "WHITESPACE_RULE",
        "MORFOLOGIK_RULE_EN_US",
        "EN_UNPAIRED_BRACKETS",
        "DASH_RULE",
        "OXFORD_COMMA"
    }

    # Filter out matches based on ruleId
    filtered_matches = [match for match in matches if match.ruleId not in excluded_rules]

    issues = [match.message for match in filtered_matches]
    return issues

def sentiment_score(text):
    return sentiment_analyzer(text)[0]

def detect_filler_words(text):
    fillers = ["um", "uh", "like", "you know", "actually", "basically", "literally"]
    found = [f for f in fillers if f in text.lower()]
    return found

def pronunciation_score(file_path, script_path=PROSODY_SCRIPT_PATH):
    score = mysp.mysppron(file_path, script_path)
    return score


def prosody_speech_rate(file_path, script_path=PROSODY_SCRIPT_PATH):
    rate = mysp.myspsr(file_path, script_path)
    return rate


def pause_analysis(file_path, min_silence_len=500, silence_thresh=-40):
    audio = AudioSegment.from_wav(file_path)
    silences = silence.detect_silence(audio,
                                      min_silence_len=min_silence_len,
                                      silence_thresh=silence_thresh)
    num_pauses = len(silences)
    durations = [(end - start) / 1000.0 for start, end in silences]
    avg_pause = sum(durations) / num_pauses if num_pauses else 0
    return num_pauses, avg_pause

def analyze_text(text, file_path):
    print("\n--- Analysis Report ---")
    print(f"\nTranscript:\n{text}\n")

    # Basic metrics
    duration = get_audio_duration(file_path)
    words = len(re.findall(r'\w+', text))
    wpm = words / (duration / 60) if duration > 0 else 0
    print(f"Speech Duration: {duration:.2f}s")
    print(f"Word Count: {words}")
    print(f"Speech Rate (WPM): {wpm:.2f}")

    # Sentiment
    sent = sentiment_score(text)
    print(f"\nSentiment: {sent['label']} (Confidence: {sent['score']:.2f})")

    # Grammar
    grammar_issues = grammar_feedback(text)
    print(f"\nGrammar Issues Found: {len(grammar_issues)}")
    for issue in grammar_issues[:3]: print(f"- {issue}")

    # Readability
    readability = textstat.flesch_reading_ease(text)
    print(f"\nReadability (Flesch): {readability:.2f}")

    # Fillers
    fillers = detect_filler_words(text)
    print(f"\nFiller Words Detected: {fillers}")

    # Pronunciation
    pron_score = pronunciation_score(file_path)
    if pron_score is not None:
        print(f"\nPronunciation Score: {pron_score:.2f}/100")
    else:
        print("\nPronunciation Score: unavailable")

    # Prosody-based speech rate
    pros_rate = prosody_speech_rate(file_path)
    if pros_rate is not None:
        print(f"Speech Rate (syllables/sec): {pros_rate:.2f}")
    else:
        print("Speech Rate (syllables/sec): unavailable")

    # Pause analysis
    num_pauses, avg_pause = pause_analysis(file_path)
    print(f"\nNumber of Pauses: {num_pauses}")
    print(f"Average Pause Duration: {avg_pause:.2f}s")

    # === Scoring ===
    grammar_score = max(0, 10 - len(grammar_issues))
    positive_score = 10 if sent['label'] == 'POSITIVE' else 5
    readability_score = 10 if readability > 60 else 5
    filler_penalty = len(fillers)
    professionalism = max(0, 10 - filler_penalty)
    total = grammar_score + positive_score + readability_score + professionalism
    print(f"\n💡 Overall Professionalism Score: {total}/40")

    # === Suggestions for Improvement ---
    print("\n--- Suggestions for Improvement ---")
    if sent['label'] == "NEGATIVE": print("⚠️ Try to sound more positive or confident.")
    if grammar_issues: print("✏️ Work on grammar; rephrase and correct tenses.")
    if readability < 60: print("📖 Make sentences clearer and easier to follow.")
    if fillers: print("🗣️ Reduce filler words to sound more professional.")
    if pros_rate is not None and pros_rate < 3: print("⏩ Try to speak a bit faster; aim for ~4–5 syll/sec.")
    if avg_pause > 1.0: print("⏸️ Minimise long pauses; aim for shorter, purposeful pauses.")

# === Run ===
if __name__ == '__main__':
    record_audio() 
    file_path = "interviewaudio.wav"  
    transcribed_text = transcribe_audio(file_path)
    if transcribed_text and not transcribed_text.startswith("Could not"):
        analyze_text(transcribed_text, file_path)
    else:
        print("Transcription failed or audio was unclear.")
