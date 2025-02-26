import streamlit as st
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification
)
import pandas as pd
from evaluate import load
import groq
import google.generativeai as genai
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime
import json
import re
import evaluate
import string
from dotenv import load_dotenv
from collections import Counter
# Import DeepEval
from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase


load_dotenv()

# Configuration and Constants
MODEL_CONFIGS = {
    "big_models": [
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemini-1.5-flash"
    ],
    "small_models": {
        "Summarization": ["bart-large-cnn", "pegasus-xsum"],
        "Translation": ["t5-base", "mbart-large-50-many-to-many-mmt"],
        "Question Answering": ["deberta-v3-large-squad2", "roberta-base-squad2"],
        "Text Classification": ["bert-base-uncased", "twitter-roberta-base-sentiment-latest"],
        "Text Generation": ["gpt2", "gpt2-medium","bart-large-cnn"],
        "Text-to-Text Generation": ["t5-small", "t5-base"]
    }
}

TASK_METRICS = {
    "Text Generation": ["perplexity", "bleu", "rouge", "meteor", "bertscore"],
    "Summarization": ["rouge", "bleu", "bertscore", "meteor",""],
    "Translation": ["bleu", "meteor", "ter", "chrf", "bertscore"],
    "Question Answering": ["f1", "answer_similarity"],
    "Text Classification": ["accuracy", "precision", "recall", "f1", "roc_auc", "matthews_correlation"],
    "Text-to-Text Generation": ["bleu", "rouge", "meteor", "bertscore"]
}

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


def initialize_apis():
    """Initialize Groq and Google Gemini APIs with environment variables."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    
    if groq_api_key:
        groq_client = groq.Groq(api_key=groq_api_key)
    else:
        st.error("Groq API key not found in environment variables!")
        groq_client = None
        
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
    else:
        st.error("Google API key not found in environment variables!")
    
    return groq_client


class MetricsCalculator:
    def __init__(self):
        self.metrics_cache = {}
        self.task_mapping = {
            "Text Generation": "text-generation",
            "Summarization": "summarization",
            "Translation": "translation",
            "Question Answering": "question-answering",
            "Text Classification": "text-classification",
            "Text-to-Text Generation": "text2text-generation"
        }

    def calculate_metrics(self, prediction: str, reference: str, task: str, input_text: str = "", small_model: str = "") -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate metrics between model outputs based on task type, returning both regular and ethical metrics."""
        results = {}
        ethical_metrics = {}
        
        try:
            # Ensure inputs are strings and properly formatted
            prediction = str(prediction).strip()
            reference = str(reference).strip()
            
            # Basic string similarity metrics for all tasks
            results['string_similarity'] = self.compute_string_similarity(prediction, reference)
            
            # Task-specific metrics
            if task in ["Text Generation", "Summarization", "Text-to-Text Generation"]:
                # ROUGE scores
                try:
                    rouge = evaluate.load("rouge")
                    rouge_scores = rouge.compute(
                        predictions=[prediction],
                        references=[reference],
                        use_stemmer=True
                    )
                    # Handle new ROUGE format
                    results.update({
                        'rouge1': round(rouge_scores['rouge1'], 4),
                        'rouge2': round(rouge_scores['rouge2'], 4),
                        'rougeL': round(rouge_scores['rougeL'], 4)
                    })
                except Exception as e:
                    st.warning(f"ROUGE calculation failed: {str(e)}")
                
                # BLEU score
                try:
                    bleu = evaluate.load("bleu")
                    bleu_score = bleu.compute(
                        predictions=[prediction],
                        references=[[reference]],
                        smooth=True
                    )
                    results['bleu'] = round(bleu_score['bleu'], 4)
                except Exception as e:
                    st.warning(f"BLEU calculation failed: {str(e)}")
                
            elif task == "Question Answering":
                results.update({
                    'exact_match': round(self.compute_exact_match(prediction, reference), 4),
                    'f1': round(self.compute_f1_score(prediction, reference), 4),
                })
                
            elif task == "Translation":
                # BLEU score for translation
                try:
                    bleu = evaluate.load("bleu")
                    bleu_score = bleu.compute(
                        predictions=[prediction],
                        references=[[reference]],
                        smooth=True
                    )
                    results['bleu'] = round(bleu_score['bleu'], 4)
                except Exception as e:
                    st.warning(f"Translation BLEU calculation failed: {str(e)}")
                
                # chrF score
                try:
                    chrf = evaluate.load("chrf")
                    chrf_score = chrf.compute(
                        predictions=[prediction],
                        references=[reference]
                    )
                    results['chrf'] = round(chrf_score['score'], 4)
                except Exception as e:
                    st.warning(f"chrF calculation failed: {str(e)}")
                
            elif task == "Text Classification":
                results['accuracy'] = round(float(prediction.strip().lower() == reference.strip().lower()), 4)
            
            # Add BERTScore for semantic similarity for all tasks
            try:
                bert_score = evaluate.load("bertscore")
                bert_scores = bert_score.compute(
                    predictions=[prediction],
                    references=[reference],
                    model_type="microsoft/deberta-xlarge-mnli",
                    batch_size=1
                )
                results['bert_score'] = round(float(bert_scores['f1'][0]), 4)
            except Exception as e:
                st.warning(f"BERTScore computation failed: {str(e)}")
            
            # DeepEval metrics (ethical testing)
            if task == "Summarization":
                try:
                    with st.spinner("Calculating ethical metrics..."):
                        # Create a test case using the input text (if available) or reference
                        test_case = LLMTestCase(
                            input=input_text if input_text else reference, 
                            actual_output=prediction
                        )
                        
                        # Initialize and run the SummarizationMetric
                        summarization_metric = SummarizationMetric(
                            threshold=0.5,
                            model=small_model,  # Use a model from MODEL_CONFIGS if appropriate
                            assessment_questions=[
                                "Is the summary accurate and free of hallucinations?",
                                "Does the summary maintain the main points of the original text?",
                                "Is the summary concise without losing important information?"
                            ]
                        )
                        
                        # Measure the metric
                        try:
                            hallucination_score = summarization_metric.measure(test_case)
                            ethical_metrics['Hallucination Score'] = round(hallucination_score, 4)
                            ethical_metrics['Hallucination Reason'] = summarization_metric.reason
                        except Exception as e:
                            st.warning(f"DeepEval summarization metric failed: {str(e)}")
                except Exception as e:
                    st.warning(f"DeepEval metrics calculation failed: {str(e)}")
            
            if not results and not ethical_metrics:
                st.warning("No metrics were successfully calculated.")
            
            return results, ethical_metrics

        except Exception as e:
            st.error(f"Error in metric calculation: {str(e)}")
            return {}, {}

    def compute_string_similarity(self, str1: str, str2: str) -> float:
        """Compute basic string similarity score."""
        from difflib import SequenceMatcher
        return round(SequenceMatcher(None, str1, str2).ratio(), 4)

    def compute_exact_match(self, prediction: str, reference: str) -> float:
        """Compute exact match score."""
        return float(self.normalize_answer(prediction) == self.normalize_answer(reference))

    def compute_f1_score(self, prediction: str, reference: str) -> float:
        """Compute F1 score."""
        prediction_tokens = self.normalize_answer(prediction).split()
        reference_tokens = self.normalize_answer(reference).split()
        
        common = Counter(prediction_tokens) & Counter(reference_tokens)
        num_same = sum(common.values())
        
        if len(prediction_tokens) == 0 or len(reference_tokens) == 0:
            return 0.0
        
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(reference_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return float(f1)

    def normalize_answer(self, text: str) -> str:
        """Normalize answer for consistent comparison."""
        text = text.lower()
        text = ''.join(ch for ch in text if ch not in string.punctuation)
        text = ' '.join(text.split())
        return text


def main():
    st.title("LLM Contextual Performance Testing")
    
    # Initialize components
    model_executor = ModelExecutor()
    metrics_calculator = MetricsCalculator()
    
    # Sidebar for model and task selection
    st.sidebar.header("Configuration")
    task = st.sidebar.selectbox("Select Task the target model is trained for", list(TASK_METRICS.keys()))
    
    # Changed: Show all big models in a single dropdown
    big_model = st.sidebar.selectbox("Select the Tester Model", MODEL_CONFIGS["big_models"])
    
    # Fixed: Properly get task-specific local models
    available_small_models = MODEL_CONFIGS["small_models"].get(task, ["bert-base-uncased"])
    small_model = st.sidebar.selectbox("Choose Model to be tested", available_small_models)
    
    # Task-specific input handling
    #st.header(f"Input Text for {task}")
    kwargs = {}
    
    if task == "Question Answering":
        context = st.text_area("Context", height=150)
        question = st.text_area("Question", height=80)
        input_text = question
        kwargs = {'context': context, 'question': question}
    elif task == "Translation":
        input_text = st.text_area("Input Text", height=150)
        src_lang = st.text_input("Source Language Code (e.g., 'en_XX')")
        tgt_lang = st.text_input("Target Language Code (e.g., 'fr_XX')")
        kwargs = {'src_lang': src_lang, 'tgt_lang': tgt_lang}
    else:
        input_text = st.text_area(f"Enter test data", height=150)
    
    # Initialize results_dict at the start
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "input": input_text if 'input_text' in locals() else "",
        "big_model": {},
        "small_model": {}
    }
    
    # Process and Compare
    if st.button("Evaluate your Model now"):
        if input_text:
            with st.spinner("Processing models..."):
                st.header("Results")
                
                # Execute Models
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Ground Truth Data")
                    big_model_output = model_executor.execute_big_model(big_model, input_text, task, **kwargs)
                    st.write(big_model_output)
                    results_dict["big_model"] = {
                        "name": big_model,
                        "output": big_model_output
                    }
                    
                with col2:
                    st.subheader("Data generated by targeted Model")
                    small_model_pipeline = model_executor.get_local_model(small_model, task)
                    if small_model_pipeline:
                        small_model_output = model_executor.execute_small_model(
                            small_model_pipeline, 
                            input_text, 
                            task, 
                            **kwargs
                        )
                        st.write(small_model_output)
                        results_dict["small_model"] = {
                            "name": small_model,
                            "output": small_model_output
                        }

                # Calculate Metrics
                if big_model_output and small_model_output:
                    st.header("Evaluation Metrics")
                    
                    with st.spinner("Calculating metrics..."):
                        try:
                            # Ensure outputs are strings
                            small_model_output = str(small_model_output)
                            big_model_output = str(big_model_output)
                            
                            # New: Get both standard and ethical metrics
                            metric_results, ethical_metric_results = metrics_calculator.calculate_metrics(
                                prediction=small_model_output,
                                reference=big_model_output,
                                task=task,
                                input_text=input_text,
                                small_model=small_model
                            )
                            
                            # Update results dictionary
                            results_dict["metrics"] = metric_results
                            results_dict["ethical_metrics"] = ethical_metric_results
                            
                            # Display standard metrics
                            if metric_results:
                                st.subheader("Standard Metric Scores")
                                metrics_df = pd.DataFrame([
                                    {"Metric": metric, "Score": f"{score:.4f}"} 
                                    for metric, score in metric_results.items()
                                ])
                                st.table(metrics_df)
                            else:
                                st.warning("No standard metrics were calculated for this task type.")
                            
                            # Display ethical metrics (in a separate table)
                            if ethical_metric_results:
                                st.subheader("Ethical Evaluation Metrics")
                                ethical_metrics_df = pd.DataFrame([
                                    {"Metric": metric, "Value": value if isinstance(value, str) else f"{value:.4f}"} 
                                    for metric, value in ethical_metric_results.items()
                                ])
                                st.table(ethical_metrics_df)
                                
                        except Exception as e:
                            st.error(f"Error calculating metrics: {str(e)}")

                    # Add download button for results
                    st.download_button(
                        "Download Results",
                        json.dumps(results_dict, indent=2),
                        f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
        else:
            st.warning("Please enter input text to compare models.")

if __name__ == "__main__":
    main()