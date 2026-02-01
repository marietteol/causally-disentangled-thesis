#!/usr/bin/env python3
"""
Unified SimCLR + Linear Evaluation on CommonVoice.
"""
import os, time, random, json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity

import torchaudio.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import os, time, random, json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity

import torchaudio.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -------------------- CONFIG --------------------
DATA_ROOT = Path('cv-corpus-23.0-2025-09-05/en')
CLIPS_DIR = DATA_ROOT / 'clips'
TSV_PATH = DATA_ROOT / 'validated.tsv'

SAMPLE_RATE = 16000
N_MELS = 64
WIN_LENGTH = 400
HOP_LENGTH = 160

BATCH_SIZE = 128
EMBED_DIM = 512
PROJ_DIM = 128
EPOCHS = 3
LR = 1e-3
TEMPERATURE = 0.07
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DURATION = 6.0
SUBSET_SPEAKERS = None

OUTPUT_DIR = Path('results_simclr')
OUTPUT_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = OUTPUT_DIR / 'evaluation_summary.json'

# Set seeds
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

RESUME = False
LAST_CHECKPOINT = OUTPUT_DIR / 'last_checkpoint.pt'

# -------------------- METADATA --------------------
metadata = pd.read_csv(TSV_PATH, sep='\t')
metadata['speaker_id'] = metadata['client_id'].astype(str)

# Standardize accent
if 'accent' in metadata.columns:
    metadata['accent'] = metadata['accent'].fillna('unknown').astype(str)
elif 'accents' in metadata.columns:
    metadata['accent'] = metadata['accents'].fillna('unknown').astype(str)
elif 'locale' in metadata.columns:
    metadata['accent'] = metadata['locale'].fillna('unknown').astype(str)
else:
    metadata['accent'] = 'unknown'

metadata['gender'] = metadata['gender'].fillna('unknown').str.lower()
metadata['age'] = metadata['age'].fillna('unknown').str.lower()

# Filter missing files
metadata['full_path'] = metadata['path'].apply(lambda p: str(CLIPS_DIR / p))
metadata = metadata[metadata['full_path'].apply(os.path.exists)]

# Collapse age
def collapse_age(age):
    if pd.isna(age): return 'unknown'
    a = str(age).lower()
    if any(x in a for x in ['teen', 'twenties']): return 'young'
    if any(x in a for x in ['thirties', 'fourties', 'fifties']): return 'adult'
    if any(x in a for x in ['sixties', 'seventies', 'eighties', 'nineties']): return 'senior'
    return 'unknown'
metadata['age_group'] = metadata['age'].apply(collapse_age)

# Map accents to 5 groups
def map_accent_to_5(acc):
    s = str(acc).lower()
    if any(k in s for k in ["united states", "american", "usa"]): return "usa"
    if any(k in s for k in ["england", "liverpool", "lancashire"]): return "england"
    if any(k in s for k in ["canada", "canadian"]): return "canada"
    if any(k in s for k in ["australia", "new zealand", "nz"]): return "australia_nz"
    if any(k in s for k in ["india", "pakistan", "sri lanka"]): return "india_south_asia"
    return "unknown"
metadata['accent_group'] = metadata['accent'].apply(map_accent_to_5)

# Optional subset
if SUBSET_SPEAKERS:
    chosen = random.sample(list(metadata['speaker_id'].unique()), SUBSET_SPEAKERS)
    metadata = metadata[metadata['speaker_id'].isin(chosen)]

# Split speakers train/val/test
speakers = metadata['speaker_id'].unique()
train_spk, test_spk = train_test_split(speakers, test_size=0.2, random_state=SEED)
val_spk, test_spk = train_test_split(test_spk, test_size=0.5, random_state=SEED)
metadata['split'] = 'train'
metadata.loc[metadata['speaker_id'].isin(val_spk), 'split'] = 'val'
metadata.loc[metadata['speaker_id'].isin(test_spk), 'split'] = 'test'

metadata.to_csv(OUTPUT_DIR / 'metadata_processed.csv', index=False)

# -------------------- DATASET --------------------

class SimCLRAudioAugment(nn.Module):
    def __init__(self,
                 noise_std=0.005,
                 gain_range=(0.8, 1.2),
                 p_noise=0.5,
                 p_gain=0.5):
        super().__init__()
        self.noise_std = noise_std
        self.gain_range = gain_range
        self.p_noise = p_noise
        self.p_gain = p_gain

    def forward(self, wav):
        # wav: (1, T)

        if torch.rand(1).item() < self.p_gain:
            gain = torch.empty(1).uniform_(*self.gain_range).item()
            wav = wav * gain

        if torch.rand(1).item() < self.p_noise:
            noise = torch.randn_like(wav) * self.noise_std
            wav = wav + noise

        return wav


class SimCLRCommonVoiceDataset(Dataset):
    def __init__(
        self,
        df,
        clips_dir,
        split='train',
        sample_rate=16000,
        duration=3.0,
        return_metadata=False
    ):
        if split is not None:
            self.df = df[df['split'] == split].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)
        self.clips_dir = clips_dir

        self.sample_rate = sample_rate   # ← FIX
        self.target_len = int(duration * sample_rate)
        self.return_metadata = return_metadata

        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.augment = SimCLRAudioAugment() if split == 'train' else None

    def __len__(self): return len(self.df)

    def load_audio(self, path):
        wav, sr = sf.read(path, dtype='float32')
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != self.sample_rate: wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        wav = torch.from_numpy(wav)
        if wav.numel() > self.target_len:
            start = torch.randint(0, wav.numel()-self.target_len+1, (1,)).item()
            wav = wav[start:start+self.target_len]
        else:
            wav = F.pad(wav, (0, self.target_len - wav.numel()))
        return wav.unsqueeze(0)

    def waveform_to_mel(self, waveform):
        spec = self.mel(waveform)
        spec = self.amplitude_to_db(spec)
        return (spec - spec.mean()) / (spec.std() + 1e-6)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
    
        # Independent crops (key SimCLR idea)
        wav1 = self.load_audio(row['full_path'])
        wav2 = self.load_audio(row['full_path'])
    
        if self.augment is not None:
            wav1 = self.augment(wav1)
            wav2 = self.augment(wav2)
    
        s1 = self.waveform_to_mel(wav1)
        s2 = self.waveform_to_mel(wav2)
    
        if self.return_metadata:
            return (
                s1.squeeze(0),
                s2.squeeze(0),
                row['speaker_id'],
                row['gender'],
                row['age_group'],
                row['accent_group']
            )
        else:
            return s1.squeeze(0), s2.squeeze(0)

def collate_fn(batch, return_metadata=False):
    s1 = torch.stack([b[0] for b in batch]).unsqueeze(1).float()
    s2 = torch.stack([b[1] for b in batch]).unsqueeze(1).float()
    if return_metadata:
        return s1, s2, [b[2] for b in batch], [b[3] for b in batch], [b[4] for b in batch], [b[5] for b in batch]
    return s1, s2

# -------------------- ONE UTTERANCE PER SPEAKER FOR TRAIN --------------------
# Keep only one random utterance per speaker in the training set
train_df = (
    metadata[metadata['split'] == 'train']
    .sort_values('full_path')  # deterministic order
    .groupby('speaker_id')
    .first()                   # always take first after sort
    .reset_index()
)

# Validation and test keep all utterances
val_df = metadata[metadata['split'] == 'val'].reset_index(drop=True)
test_df = metadata[metadata['split'] == 'test'].reset_index(drop=True)

# -------------------- CREATE DATASETS --------------------
train_ds = SimCLRCommonVoiceDataset(
    train_df,
    CLIPS_DIR,
    split=None,          # IMPORTANT
    duration=DURATION,
    return_metadata=True
)

val_ds = SimCLRCommonVoiceDataset(
    val_df,
    CLIPS_DIR,
    split=None,
    duration=DURATION,
    return_metadata=True
)

test_ds = SimCLRCommonVoiceDataset(
    test_df,
    CLIPS_DIR,
    split=None,
    duration=DURATION,
    return_metadata=True
)



# -------------------- MODEL --------------------
ce_loss = nn.CrossEntropyLoss()

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
    def forward(self, x): return self.pool(self.conv(x))

class SmallCNNEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1,32), ConvBlock(32,64), ConvBlock(64,128), ConvBlock(128,256, pool=False)
        )
        self.fc = nn.Linear(256*2, embed_dim)
    def forward(self, x):
        z = self.net(x).flatten(2)
        mean, std = z.mean(-1), z.std(-1)
        return F.normalize(self.fc(torch.cat([mean,std], dim=1)), dim=-1)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=EMBED_DIM, proj_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,in_dim), nn.BatchNorm1d(in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim,in_dim), nn.BatchNorm1d(in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim)
        )
    def forward(self, x): return self.net(x)

def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    mask = (~torch.eye(2*batch_size, 2*batch_size, dtype=torch.bool)).to(z.device)
    exp_sim = torch.exp(sim) * mask
    positives = torch.cat([torch.diag(sim,batch_size), torch.diag(sim,-batch_size)], dim=0)
    return (-torch.log(torch.exp(positives)/exp_sim.sum(dim=1))).mean()

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_):
    return GradReverse.apply(x, lambda_)

class Adversary(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.net(x)

def build_label_encoder(values):
    classes = sorted(set(values))
    return {c: i for i, c in enumerate(classes)}

gender_enc = build_label_encoder(metadata[metadata.split=="train"]["gender"])
age_enc    = build_label_encoder(metadata[metadata.split=="train"]["age_group"])
accent_enc = build_label_encoder(metadata[metadata.split=="train"]["accent_group"])

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

def lambda_schedule(epoch, lambda_max, warmup_epochs=5):
    return lambda_max * min(1.0, epoch / warmup_epochs)

def train_simclr_adversarial(
    lambda_adv_max,
    seed,
    output_dir
):
    # ---------------- Reproducibility ----------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir.mkdir(exist_ok=True)

    # ---------------- Models ----------------
    encoder = SmallCNNEncoder().to(DEVICE)
    proj    = ProjectionHead().to(DEVICE)

    gender_adv = Adversary(EMBED_DIM, len(gender_enc)).to(DEVICE)
    age_adv    = Adversary(EMBED_DIM, len(age_enc)).to(DEVICE)
    accent_adv = Adversary(EMBED_DIM, len(accent_enc)).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(proj.parameters()) +
        list(gender_adv.parameters()) +
        list(age_adv.parameters()) +
        list(accent_adv.parameters()),
        lr=LR
    )

    ce_loss = nn.CrossEntropyLoss()

    # ---------------- DataLoaders ----------------
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=lambda b: collate_fn(b, True),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda b: collate_fn(b, True)
    )


    # ---------------- History ----------------
    history = {
        "epoch": [],
        "simclr_loss": [],
        "adv_loss": [],
        "val_simclr_loss": []
    }

    # ================== Training ==================
    for epoch in range(EPOCHS):
        encoder.train()
        proj.train()
        gender_adv.train()
        age_adv.train()
        accent_adv.train()

        λ = lambda_schedule(epoch, lambda_adv_max)

        epoch_simclr_loss = 0.0
        epoch_adv_loss = 0.0

        # ---------- Train ----------
        for s1, s2, _, gender, age, accent in tqdm(
            train_loader, desc=f"λ={lambda_adv_max} | Epoch {epoch+1}"
        ):
            s1, s2 = s1.to(DEVICE), s2.to(DEVICE)

            gender_y = torch.tensor([gender_enc[g] for g in gender], device=DEVICE)
            age_y    = torch.tensor([age_enc[a] for a in age], device=DEVICE)
            acc_y    = torch.tensor([accent_enc[a] for a in accent], device=DEVICE)

            z1 = encoder(s1)
            z2 = encoder(s2)

            # -------- SimCLR --------
            p1 = F.normalize(proj(z1), dim=-1)
            p2 = F.normalize(proj(z2), dim=-1)
            simclr_loss = nt_xent_loss(p1, p2)

            # -------- Adversarial --------
            z_all = torch.cat([z1, z2], dim=0)
            z_rev = grad_reverse(z_all, λ)

            gender_y_all = torch.cat([gender_y, gender_y], dim=0)
            age_y_all    = torch.cat([age_y, age_y], dim=0)
            acc_y_all    = torch.cat([acc_y, acc_y], dim=0)

            adv_loss = (
                ce_loss(gender_adv(z_rev), gender_y_all) +
                ce_loss(age_adv(z_rev), age_y_all) +
                ce_loss(accent_adv(z_rev), acc_y_all)
            )

            loss = simclr_loss + λ * adv_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_simclr_loss += simclr_loss.item()
            epoch_adv_loss += adv_loss.item()

        # ---------- Validation ----------
        encoder.eval()
        proj.eval()

        val_simclr_loss = 0.0
        with torch.no_grad():
            for s1, s2, *_ in val_loader:
                s1, s2 = s1.to(DEVICE), s2.to(DEVICE)

                z1 = encoder(s1)
                z2 = encoder(s2)

                p1 = F.normalize(proj(z1), dim=-1)
                p2 = F.normalize(proj(z2), dim=-1)

                val_simclr_loss += nt_xent_loss(p1, p2).item()

        val_simclr_loss /= len(val_loader)

        # ---------- Store history ----------
        history["epoch"].append(epoch)
        history["simclr_loss"].append(epoch_simclr_loss / len(train_loader))
        history["adv_loss"].append(epoch_adv_loss / len(train_loader))
        history["val_simclr_loss"].append(val_simclr_loss)

        print(
            f"[λ={lambda_adv_max}] Epoch {epoch+1}/{EPOCHS} | "
            f"Train SimCLR: {history['simclr_loss'][-1]:.4f} | "
            f"Val SimCLR: {val_simclr_loss:.4f} | "
            f"Adv Loss: {history['adv_loss'][-1]:.4f} | "
            f"λ: {λ:.4f}"
        )

        # ---------- Checkpoint ----------
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "proj": proj.state_dict(),
                "gender_adv": gender_adv.state_dict(),
                "age_adv": age_adv.state_dict(),
                "accent_adv": accent_adv.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "lambda_adv": λ,
            },
            output_dir / f"checkpoint_epoch_{epoch}.pt",
        )

    return encoder, history




# -------------------- LINEAR EVALUATION --------------------

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
              seed=SEED):

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


LAMBDA_GRID = [0.2, 0.5, 1.0, 2.0, 5.0]
all_results = []

for λ in LAMBDA_GRID:
    print("\n" + "=" * 70)
    print(f"Starting run with λ_adv = {λ}")
    print("=" * 70)

    run_dir = OUTPUT_DIR / f"lambda_{λ}"
    run_dir.mkdir(exist_ok=True)

    encoder, train_history = train_simclr_adversarial(
        lambda_adv_max=λ,
        seed=SEED,
        output_dir=run_dir
    )

    print(f"\nEvaluating encoder for λ_adv = {λ} ...")

    metrics = evaluate_encoder(encoder)
    metrics["lambda_adv"] = λ
    metrics["train_history"] = train_history

    # ---- PRINT RESULTS IMMEDIATELY ----
    print(
        f"[λ={λ:.3f}] RESULTS\n"
        f"  Speaker verification:\n"
        f"    ROC-AUC : {metrics['speaker']['roc_auc']:.4f}\n"
        f"    EER     : {metrics['speaker']['eer']:.4f}\n"
        f"  Gender accuracy:\n"
        f"    Linear  : {metrics['gender']['linear']['acc']:.4f} "
        f"[{metrics['gender']['linear']['ci_lower']:.4f}, {metrics['gender']['linear']['ci_upper']:.4f}]\n"
        f"    MLP     : {metrics['gender']['mlp']['acc_mean']:.4f} ± {metrics['gender']['mlp']['acc_std']:.4f}\n"
        f"  Age accuracy:\n"
        f"    Linear  : {metrics['age']['linear']['acc']:.4f} "
        f"[{metrics['age']['linear']['ci_lower']:.4f}, {metrics['age']['linear']['ci_upper']:.4f}]\n"
        f"    MLP     : {metrics['age']['mlp']['acc_mean']:.4f} ± {metrics['age']['mlp']['acc_std']:.4f}\n"
        f"  Accent accuracy:\n"
        f"    Linear  : {metrics['accent']['linear']['acc']:.4f} "
        f"[{metrics['accent']['linear']['ci_lower']:.4f}, {metrics['accent']['linear']['ci_upper']:.4f}]\n"
        f"    MLP     : {metrics['accent']['mlp']['acc_mean']:.4f} ± {metrics['accent']['mlp']['acc_std']:.4f}"
    )
    

    # ---- SAVE IMMEDIATELY (CRASH SAFE) ----
    all_results.append(metrics)
    with open(OUTPUT_DIR / "lambda_sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
