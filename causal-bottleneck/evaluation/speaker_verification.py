def extract_branch_embeddings(dataset, encoder, bottleneck, batch_size=64):
    """
    Extracts demographic and residual embeddings for an entire dataset.

    Returns:
        z_demo, z_task, speaker_id, gender, age, accent
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    X_demo, X_task = [], []
    spk, gen, age, acc = [], [], [], []

    encoder.eval()
    bottleneck.eval()

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            speaker = batch[2]
            gender  = batch[3]
            age_g   = batch[4]
            accent  = batch[5]

            h = encoder(x)
            z_demo, z_task = bottleneck(h)

            X_demo.append(z_demo.cpu().numpy())
            X_task.append(z_task.cpu().numpy())

            spk.extend(speaker)
            gen.extend(gender)
            age.extend(age_g)
            acc.extend(accent)

    return (
        np.vstack(X_demo),
        np.vstack(X_task),
        np.array(spk),
        np.array(gen),
        np.array(age),
        np.array(acc),
    )
