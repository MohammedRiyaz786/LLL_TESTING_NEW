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
                #st.write(f"Debug: Using device {device}")
                
                # Map task to pipeline task type
                task_mapping = {
                    "Text Generation": "text-generation",
                    "Summarization": "summarization",
                    "Translation": "translation",
                    "Question Answering": "question-answering",
                    "Text Classification": "text-classification",
                    "Text-to-Text Generation": "text2text-generation"
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
                # Create task-specific system prompts
                system_prompts = {
                    "Summarization": "Summarize the following text concisely:",
                    "Translation": f"Translate the following text from {kwargs.get('src_lang', 'source language')} to {kwargs.get('tgt_lang', 'target language')}:",
                    "Question Answering": "Answer the following question based on the given context:",
                    "Text Classification": "Classify the following text:",
                    "Text-to-Text-Generation":"Generate Text based upon the prompt ",
                    "Text Generation": "Generate a coherent and contextually relevant response based on the following prompt",

                }
                
                system_prompt = system_prompts.get(task, "")
                
                if task == "Question Answering":
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
                system_prompts = {
        "Summarization": "Summarize the following text concisely:",
        "Translation": f"Translate the following text from {kwargs.get('src_lang', 'source language')} to {kwargs.get('tgt_lang', 'target language')}:",
        "Question Answering": "Answer the following question based on the given context:",
        "Text Classification": "Classify the following text in just one word :",
        "Text Generation": "Generate a coherent and contextually relevant response based on the following prompt",
    }
                system_prompt = system_prompts.get(task, "")
                
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
            if task == "Text Classification":
                result = model(input_text)
                return f"Label: {result[0]['label']}, Score: {result[0]['score']:.3f}"
                
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
                
            else:  # Text Generation
                result = model(input_text, max_length=150, min_length=30)
                return result[0]['generated_text']
                
        except Exception as e:
            st.error(f"Error executing local model: {str(e)}")
            return "Error executing model"


