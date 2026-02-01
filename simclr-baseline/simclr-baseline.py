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
from sklearn.utils.class_weight import compute_class_weight

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
EPOCHS = 30
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

RESUME = True
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

# -------------------- SPLIT DATA --------------------
# Get unique speakers
speakers = metadata['speaker_id'].unique()

# Split speakers into train/val/test
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
    """
    Dataset for SimCLR-style speaker embeddings on Common Voice.
    Generates two augmented views per audio clip and optionally returns metadata.
    """
    def __init__(self, df, clips_dir, split='train', sample_rate=16000,
                 duration=3.0, return_metadata=False):
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.clips_dir = clips_dir
        self.sample_rate = sample_rate
        self.target_len = int(duration * sample_rate)
        self.return_metadata = return_metadata

        # Mel-spectrogram transform
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=64
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        # Only apply augmentations for training
        self.augment = SimCLRAudioAugment() if split == 'train' else None

    def __len__(self):
        return len(self.df)

    def load_audio(self, path):
        """Load audio, convert to mono, resample, and pad/crop to target length."""
        wav, sr = sf.read(path, dtype='float32')
        if wav.ndim > 1:  # stereo -> mono
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        wav = torch.from_numpy(wav)

        if wav.numel() > self.target_len:
            start = torch.randint(0, wav.numel() - self.target_len + 1, (1,)).item()
            wav = wav[start:start + self.target_len]
        else:
            wav = F.pad(wav, (0, self.target_len - wav.numel()))
        return wav.unsqueeze(0)  # shape (1, T)

    def waveform_to_mel(self, waveform):
        """Convert waveform to normalized log-Mel spectrogram."""
        spec = self.mel(waveform)
        spec = self.amplitude_to_db(spec)
        return (spec - spec.mean()) / (spec.std() + 1e-6)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load two independent views of the same clip
        wav1 = self.load_audio(row['full_path'])
        wav2 = self.load_audio(row['full_path'])

        # Apply augmentations if in training
        if self.augment is not None:
            wav1 = self.augment(wav1)
            wav2 = self.augment(wav2)

        # Convert to log-Mel spectrograms
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
    """Custom collate function for DataLoader."""
    s1 = torch.stack([b[0] for b in batch]).unsqueeze(1).float()
    s2 = torch.stack([b[1] for b in batch]).unsqueeze(1).float()
    if return_metadata:
        return s1, s2, [b[2] for b in batch], [b[3] for b in batch], [b[4] for b in batch], [b[5] for b in batch]
    return s1, s2

# -------------------- MODEL --------------------
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
    """
    Convolutional encoder for log-mel spectrograms.

    Input:
        (B, 1, n_mels, time)

    Output:
        (B, embed_dim) L2-normalized embeddings
    """
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
train_dataset = SimCLRCommonVoiceDataset(train_df, CLIPS_DIR, split='train', duration=DURATION, return_metadata=False)
val_dataset   = SimCLRCommonVoiceDataset(val_df, CLIPS_DIR, split='val', duration=DURATION, return_metadata=False)
test_dataset  = SimCLRCommonVoiceDataset(test_df, CLIPS_DIR, split='test', duration=DURATION, return_metadata=True)

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

NUM_WORKERS = 8

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
    worker_init_fn=seed_worker,
    generator=g,
    collate_fn=lambda b: collate_fn(b, False),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    drop_last=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=lambda b: collate_fn(b, False),
    worker_init_fn=seed_worker,
    generator=g
)

# -------------------- TRAINING --------------------
encoder = SmallCNNEncoder().to(DEVICE)
proj    = ProjectionHead().to(DEVICE)
optimizer = torch.optim.Adam(list(encoder.parameters())+list(proj.parameters()), lr=LR)

start_epoch = 0
if RESUME and LAST_CHECKPOINT.exists():
    ckpt = torch.load(LAST_CHECKPOINT, map_location=DEVICE)
    encoder.load_state_dict(ckpt['encoder'])   
    proj.load_state_dict(ckpt['proj'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']+1
    print(f"Resumed from epoch {start_epoch}")

for epoch in range(start_epoch, EPOCHS):
    encoder.train(); proj.train(); running_loss=0
    for s1,s2 in tqdm(train_loader, desc=f"Train epoch {epoch}"):
        s1, s2 = s1.to(DEVICE), s2.to(DEVICE)
        z1, z2 = encoder(s1), encoder(s2)
        p1, p2 = F.normalize(proj(z1), dim=-1), F.normalize(proj(z2), dim=-1)
        loss = nt_xent_loss(p1,p2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch} avg loss: {running_loss/len(train_loader):.4f}")

    # validation
    encoder.eval(); proj.eval(); val_loss=0
    with torch.no_grad():
        for s1,s2 in val_loader:
            s1,s2 = s1.to(DEVICE), s2.to(DEVICE)
            z1,z2 = encoder(s1), encoder(s2)
            p1,p2 = F.normalize(proj(z1), dim=-1), F.normalize(proj(z2), dim=-1)
            val_loss += nt_xent_loss(p1,p2).item()
    print(f"Validation loss: {val_loss/len(val_loader):.4f}")

    ckpt = {'encoder':encoder.state_dict(), 'proj':proj.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    torch.save(ckpt, OUTPUT_DIR/f"checkpoint_epoch_{epoch}.pt")
    torch.save(ckpt, LAST_CHECKPOINT)

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

# -------------------- LINEAR EVALUATION --------------------
encoder.eval()
for p in encoder.parameters(): p.requires_grad=False
del proj; torch.cuda.empty_cache()

def extract_embeddings(
    dataset,
    encoder,
    batch_size=BATCH_SIZE,
    num_workers=8,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda b: collate_fn(b, True),
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

from collections import defaultdict
from itertools import combinations
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(fpr, tpr):
    fnr = 1 - tpr
    return float(fpr[np.nanargmin(np.abs(fpr - fnr))])


def tpr_at_fpr(fpr, tpr, target_fpr):
    """Return TPR at closest FPR operating point."""
    idx = np.nanargmin(np.abs(fpr - target_fpr))
    return float(tpr[idx])


def subgroup_speaker_verification(
    X,
    speakers,
    groups,
    n_pairs=20000,
    seed=0,
    far_levels=(0.001, 0.01),
):
    """
    Subgroup-conditioned speaker verification metrics.

    Args:
        X: embeddings (N, D)
        speakers: speaker IDs (N,)
        groups: dict of {group_name: np.array(N,)}
                e.g. {"gender": y_gen, "accent": y_acc}
        far_levels: operating points (FARs)

    Returns:
        nested dict with subgroup metrics + disparities
    """
    rng = np.random.default_rng(seed)
    N = len(speakers)

    # Sample global trial indices once
    idx1 = rng.integers(0, N, size=n_pairs)
    idx2 = rng.integers(0, N, size=n_pairs)

    sims = cosine_similarity(X[idx1], X[idx2]).diagonal()
    labels = (speakers[idx1] == speakers[idx2]).astype(int)

    results = {}

    for gname, gvals in groups.items():
        subgroup_metrics = {}

        for g in np.unique(gvals):
            mask = (gvals[idx1] == g) & (gvals[idx2] == g)
            if mask.sum() < 50 or len(np.unique(labels[mask])) < 2:
                continue

            y = labels[mask]
            s = sims[mask]

            auc = roc_auc_score(y, s)
            fpr, tpr, _ = roc_curve(y, s)
            eer = compute_eer(fpr, tpr)

            entry = {
                "roc_auc": float(auc),
                "eer": float(eer),
            }

            for far in far_levels:
                entry[f"tpr@far={far}"] = tpr_at_fpr(fpr, tpr, far)

            subgroup_metrics[str(g)] = entry

        # Disparities
        if subgroup_metrics:
            eer_vals = [v["eer"] for v in subgroup_metrics.values()]
            auc_vals = [v["roc_auc"] for v in subgroup_metrics.values()]

            disparities = {
                "eer_max_min": float(np.max(eer_vals) - np.min(eer_vals)),
                "eer_std": float(np.std(eer_vals)),
                "auc_max_min": float(np.max(auc_vals) - np.min(auc_vals)),
                "auc_std": float(np.std(auc_vals)),
            }
        else:
            disparities = {}

        results[gname] = {
            "subgroups": subgroup_metrics,
            "disparities": disparities,
        }

    return results


def intersectional_groups(**attrs):
    """
    Build intersectional group labels.
    Example: gender + accent → 'male|usa'
    """
    keys = list(attrs.keys())
    values = list(attrs.values())

    inter = []
    for vals in zip(*values):
        inter.append("|".join(str(v) for v in vals))

    return np.array(inter)

from collections import defaultdict
from itertools import combinations
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(fpr, tpr):
    fnr = 1 - tpr
    return float(fpr[np.nanargmin(np.abs(fpr - fnr))])


def tpr_at_fpr(fpr, tpr, target_fpr):
    """Return TPR at closest FPR operating point."""
    idx = np.nanargmin(np.abs(fpr - target_fpr))
    return float(tpr[idx])


def subgroup_speaker_verification(
    X,
    speakers,
    groups,
    n_pairs=20000,
    seed=0,
    far_levels=(0.001, 0.01),
):
    """
    Subgroup-conditioned speaker verification metrics.

    Args:
        X: embeddings (N, D)
        speakers: speaker IDs (N,)
        groups: dict of {group_name: np.array(N,)}
                e.g. {"gender": y_gen, "accent": y_acc}
        far_levels: operating points (FARs)

    Returns:
        nested dict with subgroup metrics + disparities
    """
    rng = np.random.default_rng(seed)
    N = len(speakers)

    # Sample global trial indices once
    idx1 = rng.integers(0, N, size=n_pairs)
    idx2 = rng.integers(0, N, size=n_pairs)

    sims = cosine_similarity(X[idx1], X[idx2]).diagonal()
    labels = (speakers[idx1] == speakers[idx2]).astype(int)

    results = {}

    for gname, gvals in groups.items():
        subgroup_metrics = {}

        for g in np.unique(gvals):
            mask = (gvals[idx1] == g) & (gvals[idx2] == g)
            if mask.sum() < 50 or len(np.unique(labels[mask])) < 2:
                continue

            y = labels[mask]
            s = sims[mask]

            auc = roc_auc_score(y, s)
            fpr, tpr, _ = roc_curve(y, s)
            eer = compute_eer(fpr, tpr)

            entry = {
                "roc_auc": float(auc),
                "eer": float(eer),
            }

            for far in far_levels:
                entry[f"tpr@far={far}"] = tpr_at_fpr(fpr, tpr, far)

            subgroup_metrics[str(g)] = entry

        # Disparities
        if subgroup_metrics:
            eer_vals = [v["eer"] for v in subgroup_metrics.values()]
            auc_vals = [v["roc_auc"] for v in subgroup_metrics.values()]

            disparities = {
                "eer_max_min": float(np.max(eer_vals) - np.min(eer_vals)),
                "eer_std": float(np.std(eer_vals)),
                "auc_max_min": float(np.max(auc_vals) - np.min(auc_vals)),
                "auc_std": float(np.std(auc_vals)),
            }
        else:
            disparities = {}

        results[gname] = {
            "subgroups": subgroup_metrics,
            "disparities": disparities,
        }

    return results


def intersectional_groups(**attrs):
    """
    Build intersectional group labels.
    Example: gender + accent → 'male|usa'
    """
    keys = list(attrs.keys())
    values = list(attrs.values())

    inter = []
    for vals in zip(*values):
        inter.append("|".join(str(v) for v in vals))

    return np.array(inter)


train_ds = SimCLRCommonVoiceDataset(metadata, CLIPS_DIR, split='train', duration=DURATION, return_metadata=True)
val_ds = SimCLRCommonVoiceDataset(metadata, CLIPS_DIR, split="val", duration=DURATION, return_metadata=True)
test_ds  = SimCLRCommonVoiceDataset(metadata, CLIPS_DIR, split='test',  duration=DURATION, return_metadata=True)

X_tr, y_spk_tr, y_gen_tr, y_age_tr, y_acc_tr = extract_embeddings(train_ds, encoder)
X_val, y_spk_val, y_gen_val, y_age_val, y_acc_val = extract_embeddings(val_ds, encoder)
X_te, y_spk_te, y_gen_te, y_age_te, y_acc_te = extract_embeddings(test_ds, encoder)

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

speaker_verification = speaker_verification_metrics(X_te, y_spk_te, n_pairs=30000, seed=SEED)

subgroup_verification = subgroup_speaker_verification(
    X_te,
    y_spk_te,
    groups={
        "gender": y_gen_te,
        "age_group": y_age_te,
        "accent_group": y_acc_te,
        "gender_x_accent": intersectional_groups(
            gender=y_gen_te,
            accent=y_acc_te,
        ),
    },
    n_pairs=30000,
    seed=SEED,
    far_levels=(0.001, 0.01),
)


gender_linear_val = linear_probe(X_tr, y_gen_tr, X_val, y_gen_val)
gender_linear_te  = linear_probe(X_tr, y_gen_tr, X_te,  y_gen_te)

age_linear_val = linear_probe(X_tr, y_age_tr, X_val, y_age_val)
age_linear_te  = linear_probe(X_tr, y_age_tr, X_te,  y_age_te)

accent_linear_val = linear_probe(X_tr, y_acc_tr, X_val, y_acc_val)
accent_linear_te  = linear_probe(X_tr, y_acc_tr, X_te,  y_acc_te)

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

def mlp_probe(X_tr, y_tr, X_te, y_te, hidden_dim=256, epochs=20, lr=1e-3, batch_size=256,seed=SEED):
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Encode labels
    classes, y_tr_enc = np.unique(y_tr, return_inverse=True)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_te_enc = np.array([class_to_idx.get(c, -1) for c in y_te])

    # Remove unseen test labels
    mask = y_te_enc != -1
    X_te = X_te[mask]
    y_te_enc = y_te_enc[mask]

    # Convert to tensors (keep CPU until batching)
    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr_enc).long()
    X_te_t = torch.from_numpy(X_te).float()
    y_te_t = torch.from_numpy(y_te_enc).long()

    # Class weights for imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(classes)),
        y=y_tr_enc
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    # Probe model
    probe = MLPProbe(
        in_dim=X_tr.shape[1],
        num_classes=len(classes),
        hidden_dim=hidden_dim
    ).to(DEVICE)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training
    probe.train()
    for _ in range(epochs):
        perm = torch.randperm(X_tr_t.size(0))
        for i in range(0, X_tr_t.size(0), batch_size):
            idx = perm[i:i + batch_size]

            xb = X_tr_t[idx].to(DEVICE)
            yb = y_tr_t[idx].to(DEVICE)

            logits = probe(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    probe.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, X_te_t.size(0), batch_size):
            xb = X_te_t[i:i + batch_size].to(DEVICE)
            logits = probe(xb)
            preds.append(logits.argmax(dim=1).cpu())

    preds = torch.cat(preds).numpy()

    acc = float((preds == y_te_enc).mean())
    f1  = float(f1_score(y_te_enc, preds, average="macro"))

    # Explicit cleanup
    del probe
    torch.cuda.empty_cache()

    return {
        "acc": acc,
        "f1_macro": f1
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

gender_mlp_val = probe_with_ranges(mlp_probe, X_tr, y_gen_tr, X_val, y_gen_val)
gender_mlp_te = probe_with_ranges(mlp_probe, X_tr, y_gen_tr, X_te, y_gen_te)

age_mlp_val = probe_with_ranges(mlp_probe, X_tr, y_age_tr, X_val, y_age_val)
age_mlp_te = probe_with_ranges(mlp_probe, X_tr, y_age_tr, X_te, y_age_te)

accent_mlp_val = probe_with_ranges(mlp_probe, X_tr, y_acc_tr, X_val, y_acc_val)
accent_mlp_te = probe_with_ranges(mlp_probe, X_tr, y_acc_tr, X_te, y_acc_te)



summary = {
    "speaker_verification": {
        "test": speaker_verification,
        "subgroup": subgroup_verification,
    },
    "gender": {
        "val": {
            "linear": gender_linear_val,
            "mlp": gender_mlp_val
        },
        "test": {
            "linear": gender_linear_te,
            "mlp": gender_mlp_te
        }
    },
    "age": {
        "val": {
            "linear": age_linear_val,
            "mlp": age_mlp_val
        },
        "test": {
            "linear": age_linear_te,
            "mlp": age_mlp_te
        }
    },
    "accent": {
        "val": {
            "linear": accent_linear_val,
            "mlp": accent_mlp_val
        },
        "test": {
            "linear": accent_linear_te,
            "mlp": accent_mlp_te
        }
    }
}

with open(SUMMARY_PATH, "w") as f: json.dump(summary, f, indent=2)

print("✅ Linear evaluation results saved to", SUMMARY_PATH)

print("\n========== EVALUATION RESULTS ==========\n")

print("SPEAKER VERIFICATION")
print(f"  ROC-AUC : {summary['speaker_verification']['test']['roc_auc']:.4f}")
print(f"  EER     : {summary['speaker_verification']['test']['eer']:.4f}")
print("-" * 55)

def print_task(name, res):
    print(f"{name.upper():>10}")

    for split in ["val", "test"]:
        lin = res[split]['linear']
        mlp = res[split]['mlp']

        print(f"  [{split.upper()}]")
        print(
            f"    Linear Probe Accuracy : {lin['acc']:.4f} "
            f"(CI {lin['ci_lower']:.4f}-{lin['ci_upper']:.4f})"
        )
        print(
            f"    MLP Probe Accuracy    : "
            f"{mlp['acc_mean']:.4f} ± {mlp['acc_std']:.4f} "
            f"(min={mlp['acc_min']:.4f}, max={mlp['acc_max']:.4f})"
        )

    print("-" * 55)

print_task("gender", summary["gender"])
print_task("age", summary["age"])
print_task("accent", summary["accent"])
