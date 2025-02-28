from fastapi import FastAPI, File, UploadFile
import whisper
import datetime
import subprocess
import torch
import wave
import contextlib
import numpy as np
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from transformers import pipeline

app = FastAPI()

# Load Whisper model
model_size = "large"
model = whisper.load_model(model_size)

# Load the speaker embedding model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=device)

# Load sentiment analysis model
sentiment_analyzer = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Mapping sentiment to emotion categories
def map_sentiment_to_emotion(sentiment_score):
    if sentiment_score >= 4:
        return "joy"
    elif sentiment_score == 3:
        return "neutral"
    elif sentiment_score == 2:
        return "sadness"
    else:
        return "anger"

def get_sentiment(text):
    result = sentiment_analyzer(text)[0]
    sentiment_score = int(result["label"].split()[0])  # Extract the numeric score
    return map_sentiment_to_emotion(sentiment_score)

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    filename = file.filename
    filepath = f"./{filename}"

    # Save the uploaded file
    with open(filepath, "wb") as f:
        f.write(await file.read())

    # Convert to WAV if necessary
    if not filename.endswith(".wav"):
        converted_filepath = "audio.wav"
        subprocess.call(["ffmpeg", "-i", filepath, converted_filepath, "-y"])
        filepath = converted_filepath

    # Transcribe with Whisper
    result = model.transcribe(filepath)
    segments = result["segments"]
    detected_language = result["language"]

    # Get audio duration
    with contextlib.closing(wave.open(filepath, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    # Initialize pyannote audio
    audio = Audio()

    # Function to get embeddings for each segment
    def segment_embedding(segment):
        start = segment["start"]
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(filepath, clip)
        return embedding_model(waveform[None])

    # Compute embeddings
    embeddings = np.zeros((len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment)

    # Handle NaN values
    embeddings = np.nan_to_num(embeddings)

    # Determine the number of speakers
    if len(segments) == 1:
        best_n_speakers = 1
    else:
        silhouette_scores = []
        max_num_speakers = min(10, len(segments) - 1)
        best_n_speakers = 1

        if max_num_speakers >= 2:
            for n_speakers in range(2, max_num_speakers + 1):
                clustering = AgglomerativeClustering(n_clusters=n_speakers)
                clustering.fit(embeddings)
                score = silhouette_score(embeddings, clustering.labels_)
                silhouette_scores.append((n_speakers, score))

            if silhouette_scores:
                best_n_speakers = max(silhouette_scores, key=lambda x: x[1])[0]

    # Perform speaker clustering only if multiple speakers are detected
    if best_n_speakers > 1:
        clustering = AgglomerativeClustering(n_clusters=best_n_speakers)
        labels = clustering.fit_predict(embeddings)

        for i in range(len(segments)):
            segments[i]["speaker"] = f"Speaker {labels[i] + 1}"
    else:
        for i in range(len(segments)):
            segments[i]["speaker"] = "Speaker 1"

    # Group text by speaker and perform sentiment analysis
    speaker_transcripts = {}
    for segment in segments:
        speaker = segment["speaker"]
        text = segment["text"]
        if speaker not in speaker_transcripts:
            speaker_transcripts[speaker] = ""
        speaker_transcripts[speaker] += f" {text}"

    # Compute sentiment for each speaker
    sentiment_results = {}
    for speaker, text in speaker_transcripts.items():
        sentiment_results[speaker] = get_sentiment(text)

    # Format transcript output
    transcript = f"Detected Language: {detected_language}\n"
    
    if best_n_speakers == 1:
        transcript += f"Speaker 1 (Sentiment: {sentiment_results['Speaker 1']}):\n"
        for segment in segments:
            transcript += f"{segment['text']} "
    else:
        for i, segment in enumerate(segments):
            speaker = segment["speaker"]
            if i == 0 or segments[i - 1]["speaker"] != speaker:
                transcript += f"\n{speaker} (Sentiment: {sentiment_results[speaker]}) {str(datetime.timedelta(seconds=round(segment['start'])))}: "
            transcript += f"{segment['text']}  "

    # Save transcript
    with open("transcript.txt", "w") as f:
        f.write(transcript)

    return {
        "message": "Transcription and sentiment analysis complete!",
        "detected_language": detected_language,
        "speakers": best_n_speakers,
        "sentiment_results": sentiment_results,
        "transcript": transcript
    }