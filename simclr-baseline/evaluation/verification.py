def speaker_verification_metrics(X, speakers, n_pairs=20000, seed=0):
    """
    Open-set speaker verification using cosine similarity.
    Returns ROC-AUC and EER.
    """
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

    return {
        "roc_auc": float(auc),
        "eer": float(eer)
    }
