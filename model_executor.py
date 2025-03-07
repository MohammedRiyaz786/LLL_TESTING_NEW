
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
import groq
import google.generativeai as genai
import streamlit as st
from typing import Dict, List, Tuple, Any
from api_initializer import initialize_apis

class ModelExecutor:
    def __init__(self):
        self.groq_client = initialize_apis()
        self.cached_models = {}
        self.cached_tokenizers = {}
    
    def get_local_model(self, model_name: str, task: str) -> pipeline:
        """Load and cache local models using Hugging Face pipelines with improved error handling"""
        if model_name not in self.cached_models:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if "t5" in model_name.lower():
                    pipeline_task="text2text-generation"
                    self.cached_models[model_name]=pipeline(
                    task=pipeline_task,
                    model=model_name,
                    device=device
                )
                #st.write(f"Debug: Using device {device}")
                
                # Map task to pipeline task type
                task_mapping = {
                    "Text Generation": "text-generation",
                    "Summarization": "summarization",
                    "Translation": "translation",
                    "Question Answering": "question-answering",
                    "Text Classification": "text-classification",
                    "Name Entity Recognition":"ner",
                    #"Text-to-Text Generation": "text2text-generation"
                }
                
                pipeline_task = task_mapping.get(task)
                if not pipeline_task:
                    raise ValueError(f"Unsupported task type: {task}")
                
                # Configure pipeline kwargs
                model_kwargs = {
                    "model": model_name,
                    "task": pipeline_task,
                    "device": device
                }
                
                # Special handling for translation models
                if task == "Translation":
                    model_kwargs["framework"] = "pt"
                
                self.cached_models[model_name] = pipeline(**model_kwargs)
                st.success(f"Successfully loaded model: {model_name}")
                
                return self.cached_models[model_name]
                
            except Exception as e:
                st.error(f"Error loading model {model_name}: {str(e)}")
                return None
                
        return self.cached_models[model_name]
    
    def execute_big_model(self, model_name: str, prompt: str, task: str, **kwargs) -> str:
        """Execute large models using appropriate APIs with task-specific handling."""
        try:
            if "llama" in model_name.lower() or "mixtral" in model_name.lower():
                # HIGHLIGHT: Updated system prompts to handle different classification types
                system_prompts = {
                    "Summarization": "Summarize the following text concisely:",
                    "Translation": f"Translate the following text from {kwargs.get('src_lang', 'source language')} to {kwargs.get('tgt_lang', 'target language')}:",
                    "Question Answering": "Answer the following question based on the given context:",
                    "Text Classification": "Process the following text:",
                    #"Text-to-Text-Generation":"Generate Text based upon the prompt ",
                    "Text Generation": "Generate a coherent and contextually relevant response based on the following prompt",
                    "Text Generation": "Generate a coherent and contextually relevant response based on the following prompt",
                    "Named Entity Recognition": "Extract only named entities from the following text. Return a JSON array of entity objects with 'word' and 'entity' fields. Example format: [{\"word\": \"John Smith\", \"entity\": \"PERSON\"}, {\"word\": \"New York\", \"entity\": \"LOCATION\"}]. Do not include any explanations or additional text. ",

                }
                
                system_prompt = system_prompts.get(task, "")
                
                # HIGHLIGHT: Added specific prompts for different classification types
                if task == "Text Classification":
                    classification_type = kwargs.get('classification_type', 'Sentiment Analysis')
                    if classification_type == "Sentiment Analysis":
                        system_prompt = "Classify the sentiment of the following text as either 'Positive' or 'Negative' in just one word:"
                    elif classification_type == "Spam Detection":
                        system_prompt = "Classify the following message as either 'Spam' or 'Not Spam' in just one word:"
                
                elif task == "Question Answering":
                    context = kwargs.get('context', '')
                    question = kwargs.get('question', '')
                    prompt = f"Context: {context}\nQuestion: {question}"
                elif task == "Translation":
                    src_lang = kwargs.get('src_lang', '')
                    tgt_lang = kwargs.get('tgt_lang', '')
                    prompt = f"Translate from {src_lang} to {tgt_lang}: {prompt}"
                
                completion = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=model_name,
                )
                return completion.choices[0].message.content
                
            elif "gemini" in model_name.lower():
                # HIGHLIGHT: Updated system prompts to handle different classification types
                system_prompts = {
                    "Summarization": "Summarize the following text concisely:",
                    "Translation": f"Translate the following text from {kwargs.get('src_lang', 'source language')} to {kwargs.get('tgt_lang', 'target language')}:",
                    "Question Answering": "Answer the following question based on the given context:",
                    "Text Classification": "Process the following text:",
                    "Text Generation": "Generate a coherent and contextually relevant response based on the following prompt",
                    "Named Entity Recognition": "Extract and return only named entities (e.g., persons, organizations, locations) from the following text. Output them in a structured format as a list without any explanations.Make sure not provide any explanation ",
                }
                
                system_prompt = system_prompts.get(task, "")
                
                # HIGHLIGHT: Added specific prompts for different classification types
                if task == "Text Classification":
                    classification_type = kwargs.get('classification_type', 'Sentiment Analysis')
                    if classification_type == "Sentiment Analysis":
                        system_prompt = "Classify the sentiment of the following text as either 'Positive' or 'Negative' in just one word do not tell any extra things:"
                    elif classification_type == "Spam Detection":
                        system_prompt = "Classify the following message as either 'Spam' or 'Not Spam' in just one word do not tell any extra things:"
                
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(f"{system_prompt}\n{prompt}")
                return response.text
                
            return "API Error: Could not execute model"
        except Exception as e:
            st.error(f"Error executing big model: {str(e)}")
            return "Error executing model"
    
    def execute_small_model(self, model: pipeline, input_text: str, task: str, **kwargs) -> Any:
        """Execute local models using Hugging Face pipelines with task-specific handling."""
        try:
            if task == "Text Generation":
            # Special handling for T5 models
                    if "t5" in str(model.model.config._name_or_path).lower():
                        # T5 requires specific task formatting
                        # For completion tasks with T5, we can use a custom prompt format
                        formatted_input = f"complete: {input_text}"
                        
                        result = model(
                            formatted_input,
                            max_length=50,
                            num_return_sequences=1
                        )
                        
                        # T5 output is already the generated portion only
                        if isinstance(result, list) and len(result) > 0:
                            if 'generated_text' in result[0]:
                                return result[0]['generated_text']
                            else:
                                return result[0]
                        return str(result)
                    
                    # For non-T5 models (GPT-2, BART, etc.)
                    else:
                        result = model(
                            input_text, 
                            max_length=50,
                            min_length=5,
                            do_sample=True,
                            top_k=50,
                            top_p=0.95,
                            num_return_sequences=1
                        )
                        
                        # Extract just the generated text without repeating the input
                        generated_text = result[0]['generated_text']
                        # If the generated text contains the input, try to return only the new part
                        if input_text in generated_text:
                            # Return everything after the input prompt
                            return generated_text.split(input_text, 1)[1].strip()
                        return generated_text

            if task == "Text Classification":
                result = model(input_text)
                
                # Debugging output to see the actual model response
                #st.info(f"Debug - Raw model output: {result}")
                
                # Extract label and score
                classification_type = kwargs.get('classification_type', 'Sentiment Analysis')
                
                # Different handling based on classification type
                if classification_type == "Sentiment Analysis":
                    label = result[0]['label']
                    score = result[0]['score']
                    
                    # Apply threshold for sentiment analysis
                    if label == 'positive' and score > 0.5:
                        return "Positive"
                    else:
                        return "Negative"
                        
                elif classification_type == "Spam Detection":
                    # Get the raw label - many spam detection models use 'spam'/'ham' or numeric labels
                    label = result[0]['label']
                    score = result[0]['score']
                    
                    # Check for common spam detection label patterns
                    if label.lower() == 'spam' or label == '1' or label == 1:
                        return "Spam"
                    elif label.lower() == 'ham' or label == '0' or label == 0:
                        return "Not Spam"
                    else:
                        # Fallback based on score if label is unclear
                        # Assuming higher score means more likely to be spam
                        return "Spam" if score > 0.5 else "Not Spam"
                
                # Fallback 
                return f"{label}"

                
            elif task == "Question Answering":
                context = kwargs.get('context', '')
                question = kwargs.get('question', '')
                result = model(question=question, context=context)
                return result['answer']
                
            elif task == "Summarization":
                result = model(input_text, max_length=150, min_length=30)
                return result[0]['summary_text']
                
            elif task == "Translation":
                src_lang = kwargs.get('src_lang', '')
                tgt_lang = kwargs.get('tgt_lang', '')
                result = model(input_text, src_lang=src_lang, tgt_lang=tgt_lang)
                return result[0]['translation_text']
            
            elif task == "Name Entity Recognition":
                result = model(input_text)
                
                # Format the entities for better display
                formatted_entities = []
                for entity in result:
                    formatted_entities.append({
                        "word": entity["word"],
                        "entity": entity["entity"],
                        "score": round(float(entity["score"]), 3),
                        "start": entity["start"],
                        "end": entity["end"]
                    })
                
                return formatted_entities


        except Exception as e:
            st.error(f"Error executing local model: {str(e)}")
            return "Error executing model"




