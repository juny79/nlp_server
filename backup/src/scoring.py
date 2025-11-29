from rouge import Rouge

def rouge_score(preds, labels):
    rouge = Rouge()
    return rouge.get_scores(preds, labels, avg=True)
