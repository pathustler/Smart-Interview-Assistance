import os
import re
import wave
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Core imports
import speech_recognition as sr                                       # pip install SpeechRecognition
import spacy                                                          # pip install spacy && python -m spacy download en_core_web_sm
import language_tool_python                                           # pip install language_tool_python
import textstat                                                       # pip install textstat
import time
from pydub import AudioSegment, silence                               # pip install pydub
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # pip install vaderSentiment

from record import record_audio

# Load NLP and sentiment models
import spacy.cli
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

sentiment_analyzer = SentimentIntensityAnalyzer()

# === Utility Functions ===
def get_audio_duration(path):
    """Return duration of WAV file in seconds."""
    with wave.open(path, 'rb') as wf:
        return wf.getnframes() / wf.getframerate()

# === Speech Metrics ===
def pause_metrics(path, min_silence_len=500, silence_thresh=-40):
    """Return (num_pauses, avg_pause_duration_s) via pydub silence detection."""
    try:
        audio = AudioSegment.from_wav(path)
        intervals = silence.detect_silence(
            audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
        )
        durations = [(end - start) / 1000 for start, end in intervals]
        return len(durations), (sum(durations) / len(durations)) if durations else 0
    except Exception as e:
        print(f"⚠️ Pause analysis error: {e}")
        return 0, 0

# === NLP Feature Extraction ===
def nlp_analysis(text):
    """Perform NLP analysis: POS distribution, lexical diversity, key phrases, entities."""
    doc = nlp(text)
    pos_counts = {}
    for token in doc:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    lex_div = len(set(tokens)) / len(tokens) if tokens else 0
    key_phrases = list({chunk.text.lower() for chunk in doc.noun_chunks})
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return {
        'pos_counts': pos_counts,
        'lex_diversity': lex_div,
        'key_phrases': key_phrases,
        'entities': entities
    }

# === Rule-Based Checks ===
def grammar_issues(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    excluded = {"PUNCTUATION", "WHITESPACE_RULE"}
    return [m.message for m in matches if m.ruleId not in excluded]


def detect_fillers(text):
    fillers = ["um", "uh", "like", "you know", "actually", "basically", "literally", "I mean", "Hmm", "Ah", "Er", "Ok so"]
    return [f for f in fillers if f in text.lower()]

# === Sentiment Analysis ===
def sentiment_score(text):
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores.get('compound', 0)
    label = 'POSITIVE' if compound >= 0 else 'NEGATIVE'
    return {'label': label, 'score': abs(compound)}

# === Feedback Generation ===
def generate_feedback(metrics):
    """
    Rule-based feedback generation using computed metrics.
    Returns (overall_score, strengths, weaknesses, tips)
    """
    # Scoring components
    grammar_score = max(0, 10 - len(metrics['grammar']))
    pace_score = max(0, 10 - abs(metrics['wpm'] - 140) / 14)
    diversity_score = metrics['lex_diversity'] * 10
    filler_penalty = len(metrics['fillers'])
    sentiment_score_val = 10 if metrics['sentiment']['label'] == 'POSITIVE' else 5
    total_score = round((grammar_score + pace_score + diversity_score + sentiment_score_val - filler_penalty) / 4, 1)

    strengths = []
    weaknesses = []
    tips = []

    # Strengths
    if grammar_score > 8:
        strengths.append("Strong grammar usage")
    if 120 <= metrics['wpm'] <= 160:
        strengths.append("Good speaking pace")
    if metrics['sentiment']['label'] == 'POSITIVE':
        strengths.append("Positive tone detected")
    if metrics['lex_diversity'] > 0.5:
        strengths.append("Good lexical variety")

    # Weaknesses and tips
    if len(metrics['grammar']) > 5:
        weaknesses.append(f"{len(metrics['grammar'])} grammar issues detected")
        tips.append("Review and correct grammatical errors.")
    if metrics['wpm'] < 120:
        weaknesses.append("Speaking too slowly")
        tips.append("Increase your speech rate slightly.")
    elif metrics['wpm'] > 160:
        weaknesses.append("Speaking too quickly")
        tips.append("Slow down for clarity.")
    if len(metrics['fillers']) > 2:
        weaknesses.append(f"Frequent filler words: {metrics['fillers']}")
        tips.append("Minimise filler words like 'um' and 'like'.")
    if metrics['lex_diversity'] < 0.3:
        weaknesses.append("Low lexical diversity")
        tips.append("Use a wider range of vocabulary.")
    if metrics['sentiment']['label'] == 'NEGATIVE':
        weaknesses.append("Negative tone detected")
        tips.append("Adopt a more positive and confident tone.")

    return total_score, strengths, weaknesses, tips

# === Core Analysis ===
def analyze_interview(question, audio_path):
    print("\n--- Interview Analysis Report ---")
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
    except Exception:
        text = None
    if not text:
        print("❌ Could not transcribe audio.")
        return
    print(f"\nTranscript:\n{text}\n")

    # Metrics calculation
    duration = get_audio_duration(audio_path)
    words = len(re.findall(r"\w+", text))
    wpm = words / (duration / 60) if duration > 0 else 0
    num_pauses, avg_pause = pause_metrics(audio_path)
    sentiment = sentiment_score(text)
    readability = textstat.flesch_reading_ease(text)
    grammar = grammar_issues(text)
    fillers = detect_fillers(text)
    nlpm = nlp_analysis(text)

    metrics = {
        'words': words,
        'wpm': wpm,
        'num_pauses': num_pauses,
        'avg_pause': avg_pause,
        'sentiment': sentiment,
        'readability': readability,
        'grammar': grammar,
        'fillers': fillers,
        'lex_diversity': nlpm['lex_diversity'],
        'pos_counts': nlpm['pos_counts'],
        'entities': nlpm['entities']
    }

    # Print metrics
    print(f"Duration: {duration:.1f}s | Words: {words} | WPM: {wpm:.1f}")
    print(f"Pauses: {num_pauses} avg {avg_pause:.2f}s")
    print(f"Sentiment: {sentiment['label']} ({sentiment['score']:.2f}) | Readability: {readability:.1f}")
    print(f"Grammar Issues: {len(grammar)} | Fillers: {fillers}")
    print(f"Lexical Diversity: {metrics['lex_diversity']:.2f} | POS: {metrics['pos_counts']} | Entities: {metrics['entities']}\n")

    # Generate feedback
    score, strengths, weaknesses, tips = generate_feedback(metrics)
    print(f"Overall Score: {score}/10")
    print("Strengths:")
    for s in strengths:
        print(f"- {s}")
    record_audio()
    print("Weaknesses:")
    for w in weaknesses:
        print(f"- {w}")
    print("Tips for Improvement:")
    for t in tips:
        print(f"- {t}")

# === Run ===
if __name__ == '__main__':
    #This is a placeholder, but typically, this would be replaced with a custom file with interview questions.
    questions = [
        "Tell me about yourself.",
        "Why do you want this job?"
    ]
    wait_time = 5  # seconds to wait between questions
    # Loop through each question for a full mock interview
    for idx, question in enumerate(questions, start=1):
        # Display the question and wait for user readiness
        print(f"Question {idx}/{len(questions)}: {question}")
        input("Press Enter to begin recording your answer...")
        # Record answer
        record_audio()
        # Analyze the recorded response
        analyze_interview(question, "interviewaudio.wav")
        # Prompt before next question if any remain
        if idx < len(questions):
            input("Press Enter when you're ready for the next question...")

