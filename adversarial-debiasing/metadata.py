# metadata.py
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from config import CLIPS_DIR, TSV_PATH, OUTPUT_DIR, SUBSET_SPEAKERS, SEED

def collapse_age(age):
    """Collapse raw age strings into age groups."""
    if pd.isna(age):
        return 'unknown'
    a = str(age).lower()
    if any(x in a for x in ['teen', 'twenties']):
        return 'young'
    if any(x in a for x in ['thirties', 'fourties', 'fifties']):
        return 'adult'
    if any(x in a for x in ['sixties', 'seventies', 'eighties', 'nineties']):
        return 'senior'
    return 'unknown'

def map_accent_to_5(acc):
    """Map raw accent strings into 5 standard groups."""
    s = str(acc).lower()
    if any(k in s for k in ["united states", "american", "usa"]):
        return "usa"
    if any(k in s for k in ["england", "liverpool", "lancashire"]):
        return "england"
    if any(k in s for k in ["canada", "canadian"]):
        return "canada"
    if any(k in s for k in ["australia", "new zealand", "nz"]):
        return "australia_nz"
    if any(k in s for k in ["india", "pakistan", "sri lanka"]):
        return "india_south_asia"
    return "unknown"

def load_metadata(tsv_path=TSV_PATH, clips_dir=CLIPS_DIR, subset_speakers=SUBSET_SPEAKERS, seed=SEED):
    """Load and preprocess CommonVoice metadata, returning a processed DataFrame."""
    random.seed(seed)
    
    metadata = pd.read_csv(tsv_path, sep='\t')
    metadata['speaker_id'] = metadata['client_id'].astype(str)

    # Standardize accent column
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

    # Filter out missing audio files
    metadata['full_path'] = metadata['path'].apply(lambda p: str(clips_dir / p))
    metadata = metadata[metadata['full_path'].apply(os.path.exists)]

    # Map age and accent groups
    metadata['age_group'] = metadata['age'].apply(collapse_age)
    metadata['accent_group'] = metadata['accent'].apply(map_accent_to_5)

    # Optional subset of speakers
    if subset_speakers:
        chosen = random.sample(list(metadata['speaker_id'].unique()), subset_speakers)
        metadata = metadata[metadata['speaker_id'].isin(chosen)]

    # Split speakers into train/val/test
    speakers = metadata['speaker_id'].unique()
    train_spk, test_spk = train_test_split(speakers, test_size=0.2, random_state=seed)
    val_spk, test_spk = train_test_split(test_spk, test_size=0.5, random_state=seed)

    metadata['split'] = 'train'
    metadata.loc[metadata['speaker_id'].isin(val_spk), 'split'] = 'val'
    metadata.loc[metadata['speaker_id'].isin(test_spk), 'split'] = 'test'

    # Save processed metadata
    metadata.to_csv(OUTPUT_DIR / 'metadata_processed.csv', index=False)

    return metadata
