import numpy as np
import numpy as np
import evaluate


bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, reference = eval_pred
    return bleu.compute(predictions=predictions, references=reference)