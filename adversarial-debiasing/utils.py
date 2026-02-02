import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from models import MLPProbe
import torch

def build_label_encoder(values):
    classes = sorted(set(values))
    return {c:i for i,c in enumerate(classes)}

def speaker_verification_metrics(X, speakers, n_pairs=20000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(speakers)
    idx1 = rng.integers(0, N, size=n_pairs)
    idx2 = rng.integers(0, N, size=n_pairs)
    sims = cosine_similarity(X[idx1], X[idx2]).diagonal()
    labels = (speakers[idx1] == speakers[idx2]).astype(int)
    auc = roc_auc_score(labels, sims)
    fpr, tpr, _ = roc_curve(labels, sims)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
    return {"roc_auc": float(auc), "eer": float(eer)}

def lambda_schedule(epoch, lambda_max, warmup_epochs=5):
    return lambda_max * min(1.0, epoch / warmup_epochs)

def probe_with_ranges(probe_fn, X_tr, y_tr, X_te, y_te, seeds=[0,1,2,3,4]):
    accs = []
    for s in seeds:
        res = probe_fn(X_tr, y_tr, X_te, y_te, seed=s)
        accs.append(res["acc"])
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "acc_min":  float(np.min(accs)),
        "acc_max":  float(np.max(accs))
    }
