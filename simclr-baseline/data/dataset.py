class SimCLRCommonVoiceDataset(Dataset):
    """
    CommonVoice dataset for SimCLR-style contrastive learning.

    Each item returns two independently augmented views of the same audio
    segment, as required by SimCLR.
    """
    def __init__(self, df, clips_dir, split='train', sample_rate=16000,
                 duration=3.0, return_metadata=False):
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.clips_dir = clips_dir
        self.sample_rate = sample_rate
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
