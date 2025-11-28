# placeholder for src/scoring.py
from rouge import Rouge

def rouge_score(preds, labels):
    rouge = Rouge()
    scores = rouge.get_scores(preds, labels, avg=True)
    return scores
