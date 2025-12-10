"""
SimCLR baseline

Run environment requirements:
  pip install torch torchaudio librosa numpy pandas scikit-learn umap-learn matplotlib tqdm soundfile
"""
import os
import time
import random
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import soundfile as sf
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import json

# -------- CONFIG --------
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

SUBSET_SPEAKERS = None   # None or integer to debug quickly
DURATION = 3.0           # seconds used for Mel extraction

# Closed-set probe config
CLOSED_SET_TEST_FRAC = 0.2   # fraction of utterances per train-speaker held-out for closed-set probe
MIN_UTTS_FOR_SPLIT = 2       # need >=2 utterances per speaker to split off test utterances

OUTPUT_DIR = Path('results_simclr')
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

tic = time.perf_counter()

# --------------------
# Read metadata and speaker splits
# --------------------
metadata = pd.read_csv(TSV_PATH, sep='\t')
print('Raw metadata rows:', len(metadata))

# standardize accent column
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

# keep only available audio files
metadata['full_path'] = metadata['path'].apply(lambda p: str(CLIPS_DIR / p))
metadata = metadata[metadata['full_path'].apply(lambda p: os.path.exists(p))]
print('After filtering missing files:', len(metadata))

# collapse age groups (young, adult, mature, senior)
def collapse_age(age):
    if pd.isna(age):
        return 'unknown'
    a = str(age).lower()
    if a in ['teens', 'twenties', 'teen', 'twenty']:
        return 'young'
    if a in ['thirties', 'fourties', 'thirty', 'forty']:
        return 'adult'
    if a in ['fifties', 'sixties', 'fifty', 'sixty']:
        return 'mature'
    if a in ['seventies', 'eighties', 'nineties', 'seventy', 'eighty', 'ninety']:
        return 'senior'
    return 'unknown'

metadata['age_group'] = metadata['age'].apply(collapse_age)

# mapping accents into groups (usa, england, other_uk, canada, australia_nz, india_south_asia, other)
def map_accent_to_7(acc):
    if pd.isna(acc) or str(acc).strip() == "":
        return "unknown"
    s = str(acc).lower()
    if "united states" in s or "united states english" in s or "american" in s or "us " in s or "usa" in s or "midwestern" in s:
        return "usa"
    if "england" in s or "liverpool" in s or "lancashire" in s:
        return "england"
    if "scottish" in s or "scotland" in s or "irish" in s or "northern irish" in s or "northern ireland" in s or "wales" in s or "welsh" in s:
        return "other_uk"
    if "canada" in s or "canadian" in s:
        return "canada"
    if "australia" in s or "australian" in s or "new zealand" in s or "nz" in s:
        return "australia_nz"
    if "india" in s or "south asia" in s or "pakistan" in s or "sri lanka" in s:
        return "india_south_asia"
    return "other"

metadata['accent_group'] = metadata['accent'].apply(map_accent_to_7)
metadata['speaker_id'] = metadata['client_id'].astype(str)

if SUBSET_SPEAKERS is not None:
    speakers = metadata['speaker_id'].unique().tolist()
    chosen = set(random.sample(speakers, SUBSET_SPEAKERS))
    metadata = metadata[metadata['speaker_id'].isin(chosen)]
    print('Subset rows:', len(metadata))

# create speaker-level split: train/val/test disjoint by speakers
speakers = metadata['speaker_id'].unique()
train_spk, test_spk = train_test_split(speakers, test_size=0.2, random_state=SEED)
val_spk, test_spk = train_test_split(test_spk, test_size=0.5, random_state=SEED)
print('Speakers total:', len(speakers), 'train/val/test:', len(train_spk), len(val_spk), len(test_spk))

metadata['split'] = 'train'
metadata.loc[metadata['speaker_id'].isin(val_spk), 'split'] = 'val'
metadata.loc[metadata['speaker_id'].isin(test_spk), 'split'] = 'test'

metadata.to_csv(OUTPUT_DIR / 'commonvoice_metadata_processed.csv', index=False)

# --------------------
# Audio augmentations and dataset
# --------------------
class AudioAugment:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_levels = (0.0, 0.5)
        self.pitch_shift_steps = (-2, 2)

    def random_crop_or_pad(self, waveform, target_len):
        T = waveform.shape[-1]
        if T > target_len:
            start = random.randint(0, T - target_len)
            return waveform[:, start:start + target_len]
        elif T < target_len:
            pad = target_len - T
            left = random.randint(0, pad)
            right = pad - left
            return F.pad(waveform, (left, right))
        return waveform

    def add_noise(self, waveform):
        rms = waveform.abs().mean()
        scale = random.uniform(0.0, 0.5)
        noise = torch.randn_like(waveform) * rms * scale
        return waveform + noise

    def __call__(self, waveform, target_len):
        w = self.random_crop_or_pad(waveform, target_len)
        if random.random() < 0.5:
            w = self.add_noise(w)
        return w

class SimCLRCommonVoiceDataset(Dataset):
    def __init__(self, df, clips_dir, split='train', sample_rate=16000, duration=3.0, transforms=None):
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.clips_dir = clips_dir
        self.sample_rate = sample_rate
        self.target_len = int(duration * sample_rate)
        self.transforms = transforms if transforms is not None else AudioAugment(sample_rate)
        self.mel = T.MelSpectrogram(sample_rate=sample_rate, n_fft=512, win_length=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS)
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.df)

    def load_audio(self, path):
        wav, sr = sf.read(path, dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        return torch.from_numpy(wav).unsqueeze(0)

    def waveform_to_mel(self, waveform):
        spec = self.mel(waveform)  # shape: [1, n_mels, time] when input is [1, T]
        spec = self.amplitude_to_db(spec)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        # make shape [n_mels, time]
        if spec.ndim == 3:
            spec = spec.squeeze(0)
        return spec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform = self.load_audio(row['full_path'])
        w1 = self.transforms(waveform, self.target_len)
        w2 = self.transforms(waveform, self.target_len)
        s1 = self.waveform_to_mel(w1)
        s2 = self.waveform_to_mel(w2)
        # return spectrograms and metadata
        return s1, s2, row['speaker_id'], row['gender'], row['age_group'], row['accent_group'], row.get('sentence', "")

def collate_fn(batch):
    s1 = [b[0] for b in batch]
    s2 = [b[1] for b in batch]
    s1 = torch.stack([s for s in s1], dim=0).unsqueeze(1).float()
    s2 = torch.stack([s for s in s2], dim=0).unsqueeze(1).float()
    return s1, s2, [b[2] for b in batch], [b[3] for b in batch], [b[4] for b in batch], [b[5] for b in batch], [b[6] for b in batch]

# --------------------
# Encoder & projection head
# --------------------
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

    def forward(self, x):
        return self.pool(self.conv(x))

class SmallCNNEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, embed_dim)

    def forward(self, x):
        z = self.net(x)
        z = self.pool(z).squeeze(-1).squeeze(-1)
        z = self.fc(z)
        z = F.normalize(z, dim=-1)
        return z

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=EMBED_DIM, proj_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim)
        )
    def forward(self, x):
        return self.net(x)

def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t())
    mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool)).to(z.device)
    sim = sim / temperature
    exp_sim = torch.exp(sim) * mask
    positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)
    denom = exp_sim.sum(dim=1)
    loss = -torch.log(torch.exp(positives) / denom)
    return loss.mean()

# --------------------
# DataLoaders for SimCLR training (train/val)
# --------------------
train_dataset = SimCLRCommonVoiceDataset(metadata, CLIPS_DIR, split='train', sample_rate=SAMPLE_RATE, duration=DURATION)
val_dataset = SimCLRCommonVoiceDataset(metadata, CLIPS_DIR, split='val', sample_rate=SAMPLE_RATE, duration=DURATION, transforms=AudioAugment(SAMPLE_RATE))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

encoder = SmallCNNEncoder(embed_dim=EMBED_DIM).to(DEVICE)
proj = ProjectionHead(in_dim=EMBED_DIM, proj_dim=PROJ_DIM).to(DEVICE)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=LR)

# --------------------
# Training loop (SimCLR)
# --------------------
for epoch in range(EPOCHS):
    encoder.train(); proj.train()
    running_loss = 0.0
    for s1, s2, *_ in tqdm(train_loader, desc=f'Train epoch {epoch}'):
        s1 = s1.to(DEVICE); s2 = s2.to(DEVICE)
        z1 = encoder(s1); z2 = encoder(s2)
        p1 = proj(z1); p2 = proj(z2)
        p1 = F.normalize(p1, dim=-1); p2 = F.normalize(p2, dim=-1)
        loss = nt_xent_loss(p1, p2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / max(1, len(train_loader))
    print(f'Epoch {epoch} avg loss: {avg_loss:.4f}')

    # validation loss
    encoder.eval(); proj.eval()
    val_loss = 0.0
    with torch.no_grad():
        for s1, s2, *_ in val_loader:
            s1 = s1.to(DEVICE); s2 = s2.to(DEVICE)
            z1 = encoder(s1); z2 = encoder(s2)
            p1 = proj(z1); p2 = proj(z2)
            p1 = F.normalize(p1, dim=-1); p2 = F.normalize(p2, dim=-1)
            val_loss += nt_xent_loss(p1, p2).item()
    val_loss = val_loss / max(1, len(val_loader))
    print(f'Validation loss: {val_loss:.4f}')

    torch.save({'encoder': encoder.state_dict(), 'proj': proj.state_dict(), 'epoch': epoch}, OUTPUT_DIR / f'checkpoint_epoch_{epoch}.pt')

# --------------------
# Embedding extraction (train/val/test)
# --------------------
encoder.eval()

def extract_embeddings_for_split(df_split, split_name, batch_size=128):
    ds = SimCLRCommonVoiceDataset(df_split, CLIPS_DIR, split=split_name, sample_rate=SAMPLE_RATE, duration=DURATION, transforms=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    embeddings = []
    speakers = []
    genders = []
    ages = []
    accents = []
    sentences = []
    with torch.no_grad():
        for s1, s2, spk, gen, ageg, acc, sent in tqdm(loader, desc=f'Extract {split_name}'):
            x = s1.to(DEVICE)
            z = encoder(x)
            embeddings.append(z.cpu().numpy())
            speakers.extend(spk)
            genders.extend(gen)
            ages.extend(ageg)
            accents.extend(acc)
            sentences.extend(sent)
    if len(embeddings) == 0:
        return np.zeros((0, EMBED_DIM)), [], [], [], [], []
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, speakers, genders, ages, accents, sentences

train_meta = metadata[metadata['split']=='train'].reset_index(drop=True)
val_meta = metadata[metadata['split']=='val'].reset_index(drop=True)
test_meta = metadata[metadata['split']=='test'].reset_index(drop=True)

emb_train, spk_train, gen_train, age_train, acc_train, sent_train = extract_embeddings_for_split(train_meta, 'train')
emb_val, spk_val, gen_val, age_val, acc_val, sent_val = extract_embeddings_for_split(val_meta, 'val')
emb_test, spk_test, gen_test, age_test, acc_test, sent_test = extract_embeddings_for_split(test_meta, 'test')

np.savez_compressed(OUTPUT_DIR / 'embeddings_train.npz', emb=emb_train, spk=spk_train, gen=gen_train, age=age_train, acc=acc_train)
np.savez_compressed(OUTPUT_DIR / 'embeddings_val.npz', emb=emb_val, spk=spk_val, gen=gen_val, age=age_val, acc=acc_val)
np.savez_compressed(OUTPUT_DIR / 'embeddings_test.npz', emb=emb_test, spk=spk_test, gen=gen_test, age=age_test, acc=acc_test)

# --------------------
# Utility probes & helpers
# --------------------
def train_probe(X_train, y_train, X_test, y_test, multiclass=True):
    le_train = [str(x) for x in y_train]
    le_test = [str(x) for x in y_test]
    # if there are no classes / samples return None
    if len(X_train) == 0 or len(X_test) == 0:
        return {'acc': None, 'f1_macro': None, 'clf': None}
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', multi_class='ovr', solver='saga')
    clf.fit(X_train, le_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(le_test, y_pred)
    f1 = f1_score(le_test, y_pred, average='macro')
    return {'acc': acc, 'f1_macro': f1, 'clf': clf}

def avg_by_speaker(emb, spk, label):
    agg = defaultdict(list)
    for e, s, l in zip(emb, spk, label):
        agg[s].append((e, l))
    keys = list(agg.keys())
    X = []
    Y = []
    S = []
    for k in keys:
        arr = agg[k]
        Es = np.stack([e for e,l in arr], axis=0)
        lbls = [l for e,l in arr]
        lbl = max(set(lbls), key=lbls.count)
        X.append(Es.mean(axis=0))
        Y.append(lbl)
        S.append(k)
    if len(X) == 0:
        return np.zeros((0, EMBED_DIM)), [], []
    X = np.stack(X, axis=0)
    return X, Y, S

# --------------------
# CLOSED-SET speaker-ID probe (fix): within-train per-speaker split
# --------------------
print('\n=== CLOSED-SET speaker-ID probe (using TRAIN speakers only) ===')

# Build per-speaker indices for train set (emb_train, spk_train)
idxs_by_spk_train = defaultdict(list)
for i, s in enumerate(spk_train):
    idxs_by_spk_train[s].append(i)

train_idxs = []
test_idxs = []
for spk, idxs in idxs_by_spk_train.items():
    if len(idxs) < MIN_UTTS_FOR_SPLIT:
        train_idxs.extend(idxs)
        continue
    idxs_sh = idxs.copy()
    random.shuffle(idxs_sh)
    n_test = max(1, int(len(idxs_sh) * CLOSED_SET_TEST_FRAC))
    test_part = idxs_sh[:n_test]
    train_part = idxs_sh[n_test:]
    if len(train_part) == 0:
        train_part = test_part[:-1]
        test_part = test_part[-1:]
    train_idxs.extend(train_part)
    test_idxs.extend(test_part)

X_train_utt = emb_train[np.array(train_idxs)]
y_train_utt = [spk_train[i] for i in train_idxs]

X_test_utt = emb_train[np.array(test_idxs)]
y_test_utt = [spk_train[i] for i in test_idxs]

print("Closed-set probe: #train utterances:", X_train_utt.shape[0], "#test utterances (held-out):", X_test_utt.shape[0])
print("Unique speakers in closed-set probe (train):", len(set(y_train_utt)), "held-out speakers:", len(set(y_test_utt)))
print("Intersection (should be >0):", len(set(y_train_utt).intersection(set(y_test_utt))))

res_speaker_utt = train_probe(X_train_utt, y_train_utt, X_test_utt, y_test_utt)
print("Speaker ID probe (utterance-level closed-set):", res_speaker_utt)

# Per-speaker average closed-set: average train utterances per speaker and test utterances per speaker
def avg_by_speaker_from_indices(embeddings, speakers, indices):
    agg = defaultdict(list)
    for i in indices:
        agg[speakers[i]].append(embeddings[i])
    X = []
    Y = []
    for spk, arr in agg.items():
        X.append(np.stack(arr, axis=0).mean(axis=0))
        Y.append(spk)
    if len(X) == 0:
        return np.zeros((0, EMBED_DIM)), []
    X = np.stack(X, axis=0)
    return X, Y

X_train_spk_avg, y_train_spk_avg = avg_by_speaker_from_indices(emb_train, spk_train, train_idxs)
X_test_spk_avg, y_test_spk_avg = avg_by_speaker_from_indices(emb_train, spk_train, test_idxs)

print("Closed-set per-speaker avg: #train speakers:", X_train_spk_avg.shape[0], "#test speakers:", X_test_spk_avg.shape[0])
res_speaker_spkavg = train_probe(X_train_spk_avg, y_train_spk_avg, X_test_spk_avg, y_test_spk_avg)
print("Speaker ID probe (per-speaker average closed-set):", res_speaker_spkavg)

# --------------------
# Attribute probes (gender, age, accent) using speaker-averaged embeddings
# --------------------
print('\n=== Attribute probes (train speakers -> test speakers) ===')

# Build speaker-averaged embeddings for train speakers (for training probes)
X_train_spk_attr, y_gender_train_spk, _ = avg_by_speaker(emb_train, spk_train, gen_train)
X_train_spk_age, y_age_train_spk, _ = avg_by_speaker(emb_train, spk_train, age_train)
X_train_spk_acc, y_acc_train_spk, _ = avg_by_speaker(emb_train, spk_train, acc_train)

# Build speaker-averaged embeddings for test speakers (for evaluating probes)
X_test_spk_attr, y_gender_test_spk, _ = avg_by_speaker(emb_test, spk_test, gen_test)
X_test_spk_age, y_age_test_spk, _ = avg_by_speaker(emb_test, spk_test, age_test)
X_test_spk_acc, y_acc_test_spk, _ = avg_by_speaker(emb_test, spk_test, acc_test)

res_gender = train_probe(X_train_spk_attr, y_gender_train_spk, X_test_spk_attr, y_gender_test_spk)
print('Gender probe (train-spk -> test-spk):', res_gender)
res_age = train_probe(X_train_spk_age, y_age_train_spk, X_test_spk_age, y_age_test_spk)
print('Age-group probe (train-spk -> test-spk):', res_age)
res_acc = train_probe(X_train_spk_acc, y_acc_train_spk, X_test_spk_acc, y_acc_test_spk)
print('Accent probe (train-spk -> test-spk):', res_acc)

# --------------------
# OPEN-SET speaker verification (EER)
# --------------------
print('\n=== OPEN-SET speaker verification (EER) on TEST set ===')
def build_trials_for_embeddings(emb, spk, gender, age, accent, max_pos_per_spk=50, max_neg_per_spk=50):
    idx_by_spk = defaultdict(list)
    for i, s in enumerate(spk):
        idx_by_spk[s].append(i)
    trials = []
    N = len(spk)
    all_indices = list(range(N))
    for s, idxs in idx_by_spk.items():
        if len(idxs) < 2:
            continue
        pairs = []
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                pairs.append((idxs[i], idxs[j]))
        random.shuffle(pairs)
        for a,b in pairs[:max_pos_per_spk]:
            trials.append((a,b,1, spk[a], spk[b], gender[a], gender[b], age[a], age[b], accent[a], accent[b]))
        other_indices = [i for i in all_indices if spk[i]!=s]
        sampled = random.sample(other_indices, min(max_neg_per_spk, len(other_indices)))
        for b in sampled:
            a = random.choice(idxs)
            trials.append((a,b,0, spk[a], spk[b], gender[a], gender[b], age[a], age[b], accent[a], accent[b]))
    return trials

trials = build_trials_for_embeddings(emb_test, spk_test, gen_test, age_test, acc_test, max_pos_per_spk=20, max_neg_per_spk=20)
print('Number of trials:', len(trials))

emb_test_norm = emb_test / (np.linalg.norm(emb_test, axis=1, keepdims=True) + 1e-8)
scores = []
labels = []
for a,b,lab, *rest in trials:
    sc = float(np.dot(emb_test_norm[a], emb_test_norm[b]))
    scores.append(sc)
    labels.append(lab)
scores = np.array(scores); labels = np.array(labels)

def compute_eer(labels, scores):
    fpr, tpr, thresh = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer

overall_eer = compute_eer(labels, scores)
print('Overall EER (cosine):', overall_eer)

# per-group EER (only same-group trials)
def per_group_eer(trials, labels, scores, group_name='gender'):
    groups = defaultdict(list)
    for i, t in enumerate(trials):
        a,b,lab, spk_a, spk_b, gender_a, gender_b, age_a, age_b, acc_a, acc_b = t
        if group_name=='gender':
            if gender_a==gender_b:
                groups[gender_a].append(i)
        elif group_name=='age_group':
            if age_a==age_b:
                groups[age_a].append(i)
        elif group_name=='accent_group':
            if acc_a==acc_b:
                groups[acc_a].append(i)
    results = {}
    for g, idxs in groups.items():
        if len(idxs) < 50:
            continue
        lab_g = labels[idxs]
        sc_g = scores[idxs]
        results[g] = compute_eer(lab_g, sc_g)
    return results

per_gender = per_group_eer(trials, labels, scores, 'gender')
per_age = per_group_eer(trials, labels, scores, 'age_group')
per_acc = per_group_eer(trials, labels, scores, 'accent_group')

print('EER by gender (same-group trials):', per_gender)
print('EER by age-group (same-group trials):', per_age)
print('EER by accent (same-group trials, >=50 trials):', per_acc)

# --------------------
# Save summary
# --------------------
summary = {
    'closed_set_speaker_utt_probe': {'acc': res_speaker_utt['acc'], 'f1_macro': res_speaker_utt['f1_macro']},
    'closed_set_speaker_spkavg_probe': {'acc': res_speaker_spkavg['acc'], 'f1_macro': res_speaker_spkavg['f1_macro']},
    'probe_gender_train2test': {'acc': res_gender['acc'], 'f1_macro': res_gender['f1_macro']},
    'probe_age_train2test': {'acc': res_age['acc'], 'f1_macro': res_age['f1_macro']},
    'probe_accent_train2test': {'acc': res_acc['acc'], 'f1_macro': res_acc['f1_macro']},
    'overall_eer': float(overall_eer),
    'per_gender_eer': per_gender,
    'per_age_eer': per_age,
    'per_accent_eer': per_acc,
    'closed_set_counts': {
        'closed_train_utterances': int(X_train_utt.shape[0]),
        'closed_test_utterances': int(X_test_utt.shape[0]),
        'closed_train_speakers': int(X_train_spk_avg.shape[0]),
        'closed_test_speakers': int(X_test_spk_avg.shape[0])
    }
}
with open(OUTPUT_DIR / 'evaluation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

toc = time.perf_counter()
print(f"\n Done. Results and embeddings saved under: {OUTPUT_DIR}")
print(f"⏱️ Total time: {toc - tic:0.2f} seconds")