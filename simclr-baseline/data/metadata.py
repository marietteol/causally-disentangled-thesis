"""
Metadata processing utilities for CommonVoice.
Handles accent normalization, age grouping, speaker splits.
"""

import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Split speakers train/val/test
speakers = metadata['speaker_id'].unique()
train_spk, test_spk = train_test_split(speakers, test_size=0.2, random_state=SEED)
val_spk, test_spk = train_test_split(test_spk, test_size=0.5, random_state=SEED)
metadata['split'] = 'train'
metadata.loc[metadata['speaker_id'].isin(val_spk), 'split'] = 'val'
metadata.loc[metadata['speaker_id'].isin(test_spk), 'split'] = 'test'
