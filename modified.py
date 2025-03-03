from fastapi import FastAPI, File, UploadFile
import whisper
import datetime
import subprocess
import torch
import wave
import contextlib
import numpy as np
import os
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

# Load emotion detection model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def get_emotion(text):
    result = emotion_classifier(text)[0]
    return result["label"]

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    filename = file.filename
    filepath = f"./{filename}"
    
    # Save uploaded file
    with open(filepath, "wb") as f:
        f.write(await file.read())
    
    # Always convert to mono WAV with consistent format
    converted_filepath = "processed_audio.wav"
    try:
        # Convert to mono, 16kHz WAV format (standard for most speech models)
        subprocess.call(["ffmpeg", "-i", filepath, "-ac", "1", "-ar", "16000", converted_filepath, "-y"])
        filepath = converted_filepath
    except Exception as e:
        return {"error": f"Audio conversion failed: {str(e)}"}
    
    # Transcribe with Whisper for the original language
    result = model.transcribe(filepath)
    segments = result["segments"]
    detected_language = result["language"]
    
    # Get audio duration
    with contextlib.closing(wave.open(filepath, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    
    # Initialize pyannote audio
    audio = Audio(sample_rate=16000)
    
    def segment_embedding(segment):
        try:
            start = segment["start"]
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(filepath, clip)
            
            # Ensure waveform is mono
            if len(waveform.shape) > 1 and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            elif len(waveform.shape) == 1:
                # If waveform is 1D, make it 2D (channels, samples)
                waveform = waveform.unsqueeze(0)
                
            # Handle extremely short segments
            if waveform.shape[1] < 100:  # If segment is too short
                return np.zeros(192)  # Return zero embedding
                
            return embedding_model(waveform)
        except Exception as e:
            print(f"Error processing segment {start}-{end}: {str(e)}")
            return np.zeros(192)  # Return zero embedding on error
    
    # Compute embeddings with error handling
    embeddings = np.zeros((len(segments), 192))
    for i, segment in enumerate(segments):
        try:
            embeddings[i] = segment_embedding(segment)
        except Exception as e:
            print(f"Failed to compute embedding for segment {i}: {str(e)}")
            # Keep zeros for failed embeddings
    
    # Remove rows with all zeros
    valid_indices = ~np.all(embeddings == 0, axis=1)
    valid_embeddings = embeddings[valid_indices]
    valid_segments = [seg for i, seg in enumerate(segments) if valid_indices[i]]
    
    # If we have no valid embeddings, assign all to one speaker
    if len(valid_embeddings) == 0 or len(valid_segments) == 0:
        for segment in segments:
            segment["speaker"] = "Speaker 1"
    else:
        # Determine number of speakers from valid embeddings
        best_n_speakers = 1 if len(valid_segments) == 1 else min(10, len(valid_segments))
        if best_n_speakers > 1:
            try:
                scores = []
                for n in range(2, min(best_n_speakers + 1, len(valid_embeddings))):
                    clusters = AgglomerativeClustering(n_clusters=n).fit_predict(valid_embeddings)
                    if len(set(clusters)) > 1:  # Ensure we have more than one cluster
                        score = silhouette_score(valid_embeddings, clusters)
                        scores.append((n, score))
                
                best_n_speakers = max(scores, key=lambda x: x[1])[0] if scores else 1
            except Exception as e:
                print(f"Clustering error: {str(e)}")
                best_n_speakers = 1
        
        # Speaker clustering for valid segments
        try:
            if best_n_speakers > 1 and len(valid_embeddings) >= best_n_speakers:
                labels = AgglomerativeClustering(n_clusters=best_n_speakers).fit_predict(valid_embeddings)
            else:
                labels = np.zeros(len(valid_segments))
            
            # Map labels back to original segments
            speaker_map = {}
            for i, (seg_idx, label) in enumerate(zip([i for i, valid in enumerate(valid_indices) if valid], labels)):
                speaker_map[seg_idx] = label
            
            # Assign speakers
            for i, segment in enumerate(segments):
                if i in speaker_map:
                    segment["speaker"] = f"Speaker {speaker_map[i] + 1}"
                else:
                    # For segments without valid embeddings, assign to closest valid segment
                    if speaker_map:
                        closest_seg = min(speaker_map.keys(), key=lambda x: abs(segments[x]["start"] - segment["start"]))
                        segment["speaker"] = f"Speaker {speaker_map[closest_seg] + 1}"
                    else:
                        segment["speaker"] = "Speaker 1"
        except Exception as e:
            print(f"Speaker assignment error: {str(e)}")
            # Fallback: assign all to one speaker
            for segment in segments:
                segment["speaker"] = "Speaker 1"
    
    # Rest of your code remains the same
    # Process segments for translation
    if detected_language != 'en':
        english_result = model.transcribe(filepath, task="translate")
        english_segments = english_result["segments"]
        
        for orig_segment in segments:
            closest_eng_segment = None
            min_time_diff = float('inf')
            
            for eng_segment in english_segments:
                time_diff = abs(orig_segment["start"] - eng_segment["start"]) + abs(orig_segment["end"] - eng_segment["end"])
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_eng_segment = eng_segment
            
            if closest_eng_segment:
                orig_segment["english_text"] = closest_eng_segment["text"]
            else:
                orig_segment["english_text"] = "Translation unavailable"
    
    # Prepare final transcripts with emotion and English transcription
    transcript_list = []
    english_transcript_list = []

    for segment in segments:
        speaker = segment["speaker"]
        timestamp = str(datetime.timedelta(seconds=round(segment["start"])))
        text = segment["text"]
        
        if detected_language != 'en':
            english_text = segment.get("english_text", "Translation unavailable")
        else:
            english_text = text  
    
        # Get emotion with error handling
        try:
            emotion = get_emotion(english_text)
        except Exception as e:
            print(f"Emotion detection error: {str(e)}")
            emotion = "unknown"
        
        transcript_list.append(f"{speaker} {timestamp}")
        transcript_list.append(text)
        
        english_transcript_list.append(f"{speaker} {timestamp}")
        english_transcript_list.append(english_text)
        english_transcript_list.append(f"(Emotion: {emotion})")
    
    return {
        "message": "Transcription complete!",
        "detected_language": detected_language,
        "speakers": best_n_speakers,
        "transcript": transcript_list,
        "English": english_transcript_list
    }