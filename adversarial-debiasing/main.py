# main.py
import json
from pathlib import Path
from tqdm import tqdm

import torch

from config import OUTPUT_DIR, BATCH_SIZE, DEVICE, SEED, DURATION, LAMBDA_GRID
from metadata import load_metadata
from dataset import SimCLRCommonVoiceDataset, collate_fn
from models import SmallCNNEncoder, ProjectionHead, Adversary, nt_xent_loss, grad_reverse
from utils import build_label_encoder, lambda_schedule, speaker_verification_metrics, probe_with_ranges
from evaluate import extract_embeddings, linear_probe, mlp_probe
from train import train_simclr_adversarial

def main():
    # -------------------- METADATA --------------------
    metadata = load_metadata()
    
    # -------------------- DATASETS --------------------
    train_ds = SimCLRCommonVoiceDataset(
        metadata, split='train', duration=DURATION, return_metadata=True
    )
    val_ds = SimCLRCommonVoiceDataset(
        metadata, split='val', duration=DURATION, return_metadata=False
    )
    test_ds = SimCLRCommonVoiceDataset(
        metadata, split='test', duration=DURATION, return_metadata=True
    )

    # -------------------- LABEL ENCODERS --------------------
    gender_enc = build_label_encoder(metadata[metadata.split=="train"]["gender"])
    age_enc    = build_label_encoder(metadata[metadata.split=="train"]["age_group"])
    accent_enc = build_label_encoder(metadata[metadata.split=="train"]["accent_group"])

    all_results = []

    # -------------------- LAMBDA SWEEP --------------------
    for λ in LAMBDA_GRID:
        print("\n" + "=" * 70)
        print(f"Starting run with λ_adv = {λ}")
        print("=" * 70)

        run_dir = OUTPUT_DIR / f"lambda_{λ}"
        run_dir.mkdir(exist_ok=True)

        # Train encoder
        encoder, train_history = train_simclr_adversarial(
            lambda_adv_max=λ,
            seed=SEED,
            output_dir=run_dir,
            train_ds=train_ds,
            val_ds=val_ds,
            gender_enc=gender_enc,
            age_enc=age_enc,
            accent_enc=accent_enc
        )

        print(f"\nEvaluating encoder for λ_adv = {λ} ...")

        # -------------------- EVALUATION --------------------
        metrics = evaluate_encoder(encoder, train_ds, test_ds, gender_enc, age_enc, accent_enc)

        metrics["lambda_adv"] = λ
        metrics["train_history"] = train_history

        # ---- PRINT RESULTS IMMEDIATELY ----
        print(
            f"[λ={λ:.3f}] RESULTS\n"
            f"  Speaker verification:\n"
            f"    ROC-AUC : {metrics['speaker']['roc_auc']:.4f}\n"
            f"    EER     : {metrics['speaker']['eer']:.4f}\n"
            f"  Gender accuracy:\n"
            f"    Linear  : {metrics['gender']['linear']['acc']:.4f}\n"
            f"    MLP     : {metrics['gender']['mlp']['acc_mean']:.4f} ± {metrics['gender']['mlp']['acc_std']:.4f}\n"
            f"  Age accuracy:\n"
            f"    Linear  : {metrics['age']['linear']['acc']:.4f}\n"
            f"    MLP     : {metrics['age']['mlp']['acc_mean']:.4f} ± {metrics['age']['mlp']['acc_std']:.4f}\n"
            f"  Accent accuracy:\n"
            f"    Linear  : {metrics['accent']['linear']['acc']:.4f}\n"
            f"    MLP     : {metrics['accent']['mlp']['acc_mean']:.4f} ± {metrics['accent']['mlp']['acc_std']:.4f}"
        )

        # ---- SAVE RESULTS CRASH-SAFE ----
        all_results.append(metrics)
        with open(OUTPUT_DIR / "lambda_sweep_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
