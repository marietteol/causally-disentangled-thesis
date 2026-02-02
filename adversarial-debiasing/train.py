def train_simclr_adversarial(
    lambda_adv_max,
    seed,
    output_dir
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir.mkdir(exist_ok=True)

    # Models
    encoder = SmallCNNEncoder().to(DEVICE)
    proj    = ProjectionHead().to(DEVICE)

    gender_adv  = Adversary(EMBED_DIM, len(gender_enc)).to(DEVICE)
    age_adv     = Adversary(EMBED_DIM, len(age_enc)).to(DEVICE)
    accent_adv  = Adversary(EMBED_DIM, len(accent_enc)).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(proj.parameters()) +
        list(gender_adv.parameters()) +
        list(age_adv.parameters()) +
        list(accent_adv.parameters()),
        lr=LR
    )

    ce_loss = nn.CrossEntropyLoss()

    def seed_worker(worker_id):
        worker_seed = SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    NUM_WORKERS = 8
    
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda b: collate_fn(b, True)
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda b: collate_fn(b, True)
    )

    # ---------- HISTORY ----------
    history = {
        "epoch": [],
        "simclr_loss": [],
        "adv_loss": []
    }

    # ---------------- Training ----------------
    for epoch in range(EPOCHS):
        encoder.train()
        proj.train()
        gender_adv.train()
        age_adv.train()
        accent_adv.train()

        λ = lambda_schedule(epoch, lambda_adv_max)

        epoch_simclr_loss = 0.0
        epoch_adv_loss = 0.0

        for batch_idx, (s1, s2, _, gender, age, accent) in enumerate(
            tqdm(train_loader, desc=f"λ={lambda_adv_max} | Epoch {epoch}")
        ):
            s1, s2 = s1.to(DEVICE), s2.to(DEVICE)

            gender_y = torch.tensor([gender_enc[g] for g in gender]).to(DEVICE)
            age_y    = torch.tensor([age_enc[a] for a in age]).to(DEVICE)
            acc_y    = torch.tensor([accent_enc[a] for a in accent]).to(DEVICE)

            z1 = encoder(s1)
            z2 = encoder(s2)

            # SimCLR loss
            p1 = F.normalize(proj(z1), dim=-1)
            p2 = F.normalize(proj(z2), dim=-1)
            simclr_loss = nt_xent_loss(p1, p2)

            # Concatenate embeddings from both views
            z_all = torch.cat([z1, z2], dim=0)
            
            # Gradient reversal
            z_rev = grad_reverse(z_all, λ)
            
            # Duplicate labels
            gender_y_all = torch.cat([gender_y, gender_y], dim=0)
            age_y_all    = torch.cat([age_y, age_y], dim=0)
            accent_y_all = torch.cat([acc_y, acc_y], dim=0)
            
            # Adversarial logits
            gender_logits = gender_adv(z_rev)
            age_logits    = age_adv(z_rev)
            accent_logits = accent_adv(z_rev)
            
            # Loss
            adv_loss = (
                ce_loss(gender_logits, gender_y_all) +
                ce_loss(age_logits, age_y_all) +
                ce_loss(accent_logits, accent_y_all)
            )

            loss = simclr_loss + adv_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate
            epoch_simclr_loss += simclr_loss.item()
            epoch_adv_loss += adv_loss.item()

        # Save per-epoch history
        history["epoch"].append(epoch)
        history["simclr_loss"].append(epoch_simclr_loss / len(train_loader))
        history["adv_loss"].append(epoch_adv_loss / len(train_loader))

        print(f"[λ={lambda_adv_max}] Epoch {epoch+1}/{EPOCHS} | "
              f"SimCLR Loss: {history['simclr_loss'][-1]:.4f} | "
              f"Adv Loss: {history['adv_loss'][-1]:.4f} | λ: {λ:.4f}")

        # Save checkpoint each epoch
        torch.save({
            'encoder': encoder.state_dict(),
            'proj': proj.state_dict(),
            'gender_adv': gender_adv.state_dict(),
            'age_adv': age_adv.state_dict(),
            'accent_adv': accent_adv.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'lambda_adv': λ
        }, output_dir / f"checkpoint_epoch_{epoch}.pt")

    return encoder, history
