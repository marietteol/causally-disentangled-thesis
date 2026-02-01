def train_causal_bottleneck(
    encoder,
    loader,
    k,
    num_genders,
    num_age,
    num_accent,
    lambda_gender=0.01, #default
    lambda_age=0.01, #default
    lambda_accent=0.01, #default
    lambda_orth=1.0,
    epochs=10,
    device=DEVICE,
):
    """
    Trains a linear causal bottleneck that:
    - Encodes demographic information in z_demo
    - Removes demographic information from z_task via adversarial GRL
    - Enforces orthogonality between branches

    Args:
        encoder: frozen feature extractor
        loader: training DataLoader
        k: bottleneck dimensionality
        lambda_*: adversarial strengths
        lambda_orth: orthogonality regularization
        epochs: number of training epochs
    """
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    bottleneck = LinearCausalBottleneck(encoder.output_dim, k=k).to(device)

    gender_clf = nn.Linear(k, num_genders).to(device)
    age_clf    = nn.Linear(k, num_age).to(device)
    accent_clf = nn.Linear(k, num_accent).to(device)

    gender_adv = nn.Linear(encoder.output_dim, num_genders).to(device)
    age_adv    = nn.Linear(encoder.output_dim, num_age).to(device)
    accent_adv = nn.Linear(encoder.output_dim, num_accent).to(device)

    CE = nn.CrossEntropyLoss()

    opt_main = torch.optim.Adam(
        list(bottleneck.parameters()) +
        list(gender_clf.parameters()) +
        list(age_clf.parameters()) +
        list(accent_clf.parameters()),
        lr=1e-3,
    )

    opt_adv = torch.optim.Adam(
        list(gender_adv.parameters()) +
        list(age_adv.parameters()) +
        list(accent_adv.parameters()),
        lr=1e-3,
    )

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in loader:
            x = batch[0].to(device)
            gender = batch[3].to(device)
            age    = batch[4].to(device)
            accent = batch[5].to(device)

            h = encoder(x)
            z_demo, z_task = bottleneck(h, detach_residual=False)

            # --- Adversary update ---
            opt_adv.zero_grad()
            adv_loss = (
                CE(gender_adv(z_task.detach()), gender) +
                CE(age_adv(z_task.detach()), age) +
                CE(accent_adv(z_task.detach()), accent)
            )
            adv_loss.backward()
            opt_adv.step()

            # --- Main update ---
            opt_main.zero_grad()

            demo_loss = (
                CE(gender_clf(z_demo), gender) +
                CE(age_clf(z_demo), age) +
                CE(accent_clf(z_demo), accent)
            )

            grl_loss = (
                CE(gender_adv(grad_reverse(z_task, lambda_gender)), gender) +
                CE(age_adv(grad_reverse(z_task, lambda_age)), age) +
                CE(accent_adv(grad_reverse(z_task, lambda_accent)), accent)
            )

            orth_loss = lambda_orth * orthogonality_loss(z_demo, z_task)

            loss = demo_loss + grl_loss + orth_loss
            loss.backward()
            opt_main.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss / len(loader):.4f}")

    return bottleneck
