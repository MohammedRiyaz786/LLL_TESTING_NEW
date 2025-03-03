from fastapi import FastAPI, File, UploadFile
import whisper
import datetime
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
    
    # Transcribe with Whisper for the original language
    try:
        result = model.transcribe(filepath)
        segments = result["segments"]
        detected_language = result["language"]
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}
    
    # Get audio duration
    try:
        if filepath.lower().endswith('.wav'):
            with contextlib.closing(wave.open(filepath, "r")) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
        else:
            # For non-WAV files, use whisper's result
            duration = segments[-1]["end"] if segments else 0
    except Exception as e:
        # Fallback to estimated duration from segments
        duration = segments[-1]["end"] if segments else 0
    
    # Initialize pyannote audio
    audio = Audio()
    
    def segment_embedding(segment):
        try:
            start = segment["start"]
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(filepath, clip)
            
            # Handle multi-channel audio - convert to mono
            if len(waveform.shape) > 1 and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Handle 1D tensors by converting to 2D
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            
            # Skip segments that are too short
            if waveform.shape[1] < 100:
                return np.zeros(192)
                
            # Process the embedding
            return embedding_model(waveform)
        except Exception as e:
            print(f"Error in segment_embedding: {str(e)}")
            return np.zeros(192)  # Return zeros on error
    
    # Compute embeddings
    embeddings = np.zeros((len(segments), 192))
    valid_segments = []
    valid_embeddings = []
    
    for i, segment in enumerate(segments):
        try:
            embedding = segment_embedding(segment)
            embeddings[i] = embedding
            
            # Track valid embeddings for clustering
            if not np.all(embedding == 0):
                valid_segments.append(segment)
                valid_embeddings.append(embedding)
        except Exception as e:
            print(f"Failed to compute embedding for segment {i}: {str(e)}")
    
    # Handle case with no valid embeddings
    if not valid_embeddings:
        for segment in segments:
            segment["speaker"] = "Speaker 1"
        best_n_speakers = 1
    else:
        # Convert to numpy array
        valid_embeddings = np.array(valid_embeddings)
        
        # Determine number of speakers
        best_n_speakers = 1
        if len(valid_embeddings) > 1:
            try:
                # Try clustering with different numbers of speakers
                max_speakers = min(10, len(valid_embeddings))
                best_score = -1
                
                for n_speakers in range(2, max_speakers + 1):
                    # Skip if we don't have enough samples
                    if n_speakers >= len(valid_embeddings):
                        continue
                        
                    clustering = AgglomerativeClustering(n_clusters=n_speakers)
                    labels = clustering.fit_predict(valid_embeddings)
                    
                    # Need at least 2 samples per cluster for silhouette score
                    if len(set(labels)) < 2 or min(np.bincount(labels)) < 2:
                        continue
                        
                    score = silhouette_score(valid_embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_n_speakers = n_speakers
                
                # Default to 1 speaker if clustering fails
                if best_score < 0:
                    best_n_speakers = 1
            except Exception as e:
                print(f"Clustering error: {str(e)}")
                best_n_speakers = 1
        
        # Perform final clustering
        try:
            if best_n_speakers > 1:
                clustering = AgglomerativeClustering(n_clusters=best_n_speakers)
                valid_labels = clustering.fit_predict(valid_embeddings)
                
                # Map valid segments back to all segments
                valid_indices = [segments.index(seg) for seg in valid_segments]
                speaker_map = {}
                
                for idx, label in zip(valid_indices, valid_labels):
                    speaker_map[idx] = label
                
                # Assign speakers to all segments
                for i, segment in enumerate(segments):
                    if i in speaker_map:
                        segment["speaker"] = f"Speaker {speaker_map[i] + 1}"
                    else:
                        # Find nearest valid segment by time
                        if valid_indices:
                            nearest_idx = min(valid_indices, 
                                             key=lambda x: abs(segments[x]["start"] - segment["start"]))
                            segment["speaker"] = f"Speaker {speaker_map[nearest_idx] + 1}"
                        else:
                            segment["speaker"] = "Speaker 1"
            else:
                # Only one speaker
                for segment in segments:
                    segment["speaker"] = "Speaker 1"
        except Exception as e:
            print(f"Speaker assignment error: {str(e)}")
            # Fallback to single speaker
            for segment in segments:
                segment["speaker"] = "Speaker 1"
    
    # Process segments for translation - using audio file approach
    if detected_language != 'en':
        try:
            # Run a separate transcription in English/translation mode to get parallel segments
            english_result = model.transcribe(filepath, task="translate")
            english_segments = english_result["segments"]
            
            # Map translated segments to original segments 
            for orig_segment in segments:
                closest_eng_segment = None
                min_time_diff = float('inf')
                
                # Find the English segment with the closest start time
                for eng_segment in english_segments:
                    time_diff = abs(orig_segment["start"] - eng_segment["start"]) + abs(orig_segment["end"] - eng_segment["end"])
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_eng_segment = eng_segment
                
                # Store the matching English text in the original segment
                if closest_eng_segment:
                    orig_segment["english_text"] = closest_eng_segment["text"]
                else:
                    # Fallback 
                    orig_segment["english_text"] = "Translation unavailable"
        except Exception as e:
            print(f"Translation error: {str(e)}")
            # Fallback - mark all segments with translation unavailable
            for segment in segments:
                segment["english_text"] = "Translation unavailable"
    
    # Prepare final transcripts with emotion and English transcription
    transcript_list = []
    english_transcript_list = []

    for segment in segments:
        speaker = segment["speaker"]
        timestamp = str(datetime.timedelta(seconds=round(segment["start"])))
        text = segment["text"]
        
        # Get English text 
        if detected_language != 'en':
            english_text = segment.get("english_text", "Translation unavailable")
        else:
            english_text = text  
    
        # Emotion detection on the English transcription
        try:
            emotion = get_emotion(english_text)
        except Exception as e:
            print(f"Emotion detection error: {str(e)}")
            emotion = "neutral"  # fallback
        
        # Append to transcript list for detected language
        transcript_list.append(f"{speaker} {timestamp}")
        transcript_list.append(text)
        
        # Append to English transcript list for English translation with emotion
        english_transcript_list.append(f"{speaker} {timestamp}")
        english_transcript_list.append(english_text)
        english_transcript_list.append(f"(Emotion: {emotion})")
    
    # Clean up
    try:
        os.remove(filepath)
    except:
        pass
    
    return {
        "message": "Transcription complete!",
        "detected_language": detected_language,
        "speakers": best_n_speakers,
        "transcript": transcript_list,
        "English": english_transcript_list
    }