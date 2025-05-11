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
from record import record_audio


# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

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

def analyze_text(text):
    print("\n--- Analysis Report ---")
    print(f"\nTranscript:\n{text}\n")

    # Sentiment
    sentiment = sentiment_score(text)
    print(f"Sentiment: {sentiment['label']} (Confidence: {sentiment['score']:.2f})")

    # Grammar
    grammar_issues = grammar_feedback(text)
    print(f"\nGrammar Issues Found: {len(grammar_issues)}")
    for issue in grammar_issues[:3]:
        print(f"- {issue}")

    # Readability
    readability_score = textstat.flesch_reading_ease(text)
    print(f"\nReadability (Flesch): {readability_score:.2f}")

    # Fillers
    filler_words = detect_filler_words(text)
    print(f"\nFiller Words Detected: {filler_words}")

    # Word count
    word_count = len(re.findall(r'\w+', text))
    print(f"\nWord Count: {word_count}")

    # === Scoring ===
    grammar_score = max(0, 10 - len(grammar_issues))
    sentiment_score_val = 10 if sentiment['label'] == 'POSITIVE' else 5
    readability_score_val = 10 if readability_score > 60 else 5
    filler_penalty = len(filler_words) * 1
    professionalism_score = max(0, 10 - filler_penalty)

    total_score = grammar_score + sentiment_score_val + readability_score_val + professionalism_score
    print(f"\n💡 Overall Professionalism Score: {total_score}/40")

    # === Feedback ===
    print("\n--- Suggestions for Improvement ---")
    if sentiment['label'] == "NEGATIVE":
        print("⚠️ Try to sound more positive or confident in your responses.")
    if grammar_issues:
        print("✏️ Work on grammar. Consider rephrasing sentences or using correct tenses.")
    if readability_score < 60:
        print("📖 Make your responses clearer and easier to follow.")
    if filler_words:
        print("🗣️ Reduce filler words like 'um', 'like', etc. to sound more professional.")

# === Run ===
record_audio() 
file_path = "interviewaudio.wav"  
transcribed_text = transcribe_audio(file_path)
if transcribed_text and not transcribed_text.startswith("Could not"):
    analyze_text(transcribed_text)
else:
    print("Transcription failed or audio was unclear.")
