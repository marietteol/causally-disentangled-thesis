def extract_embeddings(dataset, encoder):
    def seed_worker(worker_id):
        worker_seed = SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        collate_fn=lambda b: collate_fn(b, True)
    )

    X, spk, gen, age, acc = [], [], [], [], []
    encoder.eval()

    with torch.no_grad():
        for s1, _, speaker, gender, age_g, accent in tqdm(
            loader, desc="Extract embeddings"
        ):
            s1 = s1.to(DEVICE, non_blocking=True)
            z = encoder(s1).cpu().numpy()

            X.append(z)
            spk.extend(speaker)
            gen.extend(gender)
            age.extend(age_g)
            acc.extend(accent)

    return (
        np.vstack(X),
        np.array(spk),
        np.array(gen),
        np.array(age),
        np.array(acc),
    )

def linear_probe(X_tr, y_tr, X_te, y_te, n_bootstrap=1000, ci=95, random_state=0):
    rng = np.random.RandomState(random_state)

    clf = LogisticRegression(max_iter=1000, n_jobs=1, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    # Point estimate
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    # Bootstrap CI
    n = len(y_te)
    boot_accs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_acc = accuracy_score(y_te[idx], y_pred[idx])
        boot_accs.append(boot_acc)

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_accs, alpha)
    upper = np.percentile(boot_accs, 100 - alpha)

    return {
        "acc": acc,
        "ci_lower": lower,
        "ci_upper": upper,
    }

class MLPProbe(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def mlp_probe(X_tr, y_tr, X_te, y_te,
              hidden_dim=256,
              epochs=20,
              lr=1e-3,
              batch_size=256,
              seed=0):

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Label encoding
    classes, y_tr_enc = np.unique(y_tr, return_inverse=True)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_te_enc = np.array([class_to_idx.get(c, -1) for c in y_te])

    mask = y_te_enc != -1
    X_te = X_te[mask]
    y_te_enc = y_te_enc[mask]

    X_tr_t = torch.from_numpy(X_tr).float().to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr_enc).long().to(DEVICE)
    X_te_t = torch.from_numpy(X_te).float().to(DEVICE)
    y_te_t = torch.from_numpy(y_te_enc).long().to(DEVICE)

    probe = MLPProbe(
        in_dim=X_tr.shape[1],
        num_classes=len(classes),
        hidden_dim=hidden_dim
    ).to(DEVICE)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(epochs):
        perm = torch.randperm(X_tr_t.size(0))
        for i in range(0, X_tr_t.size(0), batch_size):
            idx = perm[i:i+batch_size]
            logits = probe(X_tr_t[idx])
            loss = criterion(logits, y_tr_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation (Top-1 accuracy)
    probe.eval()
    with torch.no_grad():
        preds = probe(X_te_t).argmax(dim=1)

    acc = (preds.cpu().numpy() == y_te_enc).mean()

    return {
        "acc": float(acc)
    }

def evaluate_encoder(encoder):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    X_tr, y_spk_tr, y_gen_tr, y_age_tr, y_acc_tr = extract_embeddings(train_ds, encoder)
    X_te, y_spk_te, y_gen_te, y_age_te, y_acc_te = extract_embeddings(test_ds, encoder)

    return {
        "speaker": speaker_verification_metrics(X_te, y_spk_te, seed=SEED),
        "gender": {
            "linear": linear_probe(X_tr, y_gen_tr, X_te, y_gen_te),
            "mlp": probe_with_ranges(mlp_probe, X_tr, y_gen_tr, X_te, y_gen_te)
        },
        "age": {
            "linear": linear_probe(X_tr, y_age_tr, X_te, y_age_te),
            "mlp": probe_with_ranges(mlp_probe, X_tr, y_age_tr, X_te, y_age_te)
        },
        "accent": {
            "linear": linear_probe(X_tr, y_acc_tr, X_te, y_acc_te),
            "mlp": probe_with_ranges(mlp_probe, X_tr, y_acc_tr, X_te, y_acc_te)
        }
    }
