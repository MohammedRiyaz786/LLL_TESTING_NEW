from collections import Counter
import string
import evaluate
import streamlit  as st
from difflib import SequenceMatcher

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

    def calculate_metrics(self, prediction: str, reference: str, task: str) -> Dict[str, float]:
        """Calculate metrics between model outputs based on task type."""
        results = {}
        
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
            
            if not results:
                st.warning("No metrics were successfully calculated.")
            
            return results

        except Exception as e:
            st.error(f"Error in metric calculation: {str(e)}")
            return {}

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

