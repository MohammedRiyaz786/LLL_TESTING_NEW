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
         "Text Classification": {
            "Sentiment Analysis": ["twitter-roberta-base-sentiment-latest"],
            "Spam Detection": ["bert-tiny-finetuned-sms-spam-detection"]
        },
        # "Text Classification": ["bert-base-uncased", "twitter-roberta-base-sentiment-latest"],
        "Text Generation": ["bart-large-cnn","t5-small", "t5-base"],
        "Name Entity Recognition":["roberta-large-ner-english"],
        #"Text-to-Text Generation": ["t5-small", "t5-base"]
    }
}

TASK_METRICS = {
    "Text Generation": ["perplexity", "bleu", "rouge", "meteor", "bertscore"],
    "Summarization": ["rouge", "bleu", "bertscore", "meteor",""],
    "Translation": ["bleu", "meteor", "ter", "chrf", "bertscore"],
    "Question Answering": ["f1", "answer_similarity"],
    "Named Entity Recognition": ['precision', 'recall', 'f1'],
    "Text Classification": ["accuracy", "precision", "recall", "f1", "roc_auc", "matthews_correlation"],
    #"Text-to-Text Generation": ["bleu", "rouge", "meteor", "bertscore"]
}
