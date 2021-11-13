import pickle
from sklearn.metrics import accuracy_score, f1_score

def dump_pickled_data(path: str, data: object):
    """Dump data"""
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
    except FileNotFoundError:
        print("Directory is not available.")
        raise
        
def eval_metrics(y_true, y_preds):
    """
    Evaluation function
    """
    acc = accuracy_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds, average="macro")
    
    return acc, f1