from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import pipeline


model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print(classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers."))