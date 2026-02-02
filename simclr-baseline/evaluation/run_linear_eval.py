from config import *
from data.metadata import process_commonvoice_metadata
from data.dataset import SimCLRCommonVoiceDataset
from models.encoder import SmallCNNEncoder
from evaluation.probes import *
from evaluation.verification import speaker_verification_metrics

encoder.eval()
for p in encoder.parameters(): p.requires_grad=False
del proj; torch.cuda.empty_cache()

train_ds = SimCLRCommonVoiceDataset(metadata, CLIPS_DIR, split='train', duration=DURATION, return_metadata=True)
test_ds  = SimCLRCommonVoiceDataset(metadata, CLIPS_DIR, split='test',  duration=DURATION, return_metadata=True)

X_tr, y_spk_tr, y_gen_tr, y_age_tr, y_acc_tr = extract_embeddings(train_ds, encoder)
X_te, y_spk_te, y_gen_te, y_age_te, y_acc_te = extract_embeddings(test_ds, encoder)

speaker_verification = speaker_verification_metrics(X_te, y_spk_te, n_pairs=30000, seed=SEED)
gender_linear  = linear_probe(X_tr, y_gen_tr, X_te, y_gen_te)
age_linear     = linear_probe(X_tr, y_age_tr, X_te, y_age_te)
accent_linear  = linear_probe(X_tr, y_acc_tr, X_te, y_acc_te)

gender_mlp = probe_with_ranges(
    mlp_probe, X_tr, y_gen_tr, X_te, y_gen_te
)

age_mlp = probe_with_ranges(
    mlp_probe, X_tr, y_age_tr, X_te, y_age_te
)

accent_mlp = probe_with_ranges(
    mlp_probe, X_tr, y_acc_tr, X_te, y_acc_te
)

summary = {
    "speaker_verification": speaker_verification,
    "gender": {
        "linear": gender_linear,
        "mlp": gender_mlp
    },
    "age": {
        "linear": age_linear,
        "mlp": age_mlp
    },
    "accent": {
        "linear": accent_linear,
        "mlp": accent_mlp
    }
}

with open(SUMMARY_PATH, "w") as f: json.dump(summary, f, indent=2)
