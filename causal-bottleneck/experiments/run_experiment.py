def run_experiments(
    encoder,
    train_loader,
    train_ds,
    val_ds,
    test_ds,
    k_values,
    lambda_grid,
    num_genders,
    num_age,
    num_accent,
    epochs=10,
    device=DEVICE,
    seed=SEED,
):
    """
    Runs causal bottleneck experiments over bottleneck sizes and
    adversarial strength configurations.

    Returns:
        List of experiment result dictionaries.
    """
    results = []

    for k in k_values:
        for lambdas in lambda_grid:
            print("\n" + "=" * 80)
            print(f"Training bottleneck | k={k} | lambda_adv={lambdas}")
            print("=" * 80)

            # --------------------------------------------------
            # Train causal bottleneck
            # --------------------------------------------------
            bottleneck = train_push_demographics_fixed(
                encoder=encoder,
                loader=train_loader,
                k=k,
                num_genders=num_genders,
                num_age=num_age,
                num_accent=num_accent,
                lambda_gender=lambdas["gender"],
                lambda_age=lambdas["age"],
                lambda_accent=lambdas["accent"],
                lambda_orth=1.0,
                epochs=epochs,
                device=device,
            )

            # --------------------------------------------------
            # Extract embeddings (FULL datasets)
            # --------------------------------------------------
            X_demo_tr, X_task_tr, y_spk_tr, y_gen_tr, y_age_tr, y_acc_tr = \
                extract_branch_embeddings(train_ds, encoder, bottleneck)

            X_demo_val, X_task_val, _, y_gen_val, y_age_val, y_acc_val = \
                extract_branch_embeddings(val_ds, encoder, bottleneck)

            X_demo_te, X_task_te, y_spk_te, y_gen_te, y_age_te, y_acc_te = \
                extract_branch_embeddings(test_ds, encoder, bottleneck)

            # --------------------------------------------------
            # Speaker verification (task branch only)
            # --------------------------------------------------
            sv_metrics = speaker_verification_metrics(
                X_task_te,
                y_spk_te,
                n_pairs=min(5000, len(y_spk_te)),
                seed=seed,
            )

            # --------------------------------------------------
            # Demographic leakage probes
            # --------------------------------------------------
            demo_acc = {}

            attributes = {
                "gender": (y_gen_tr, y_gen_val, y_gen_te),
                "age":    (y_age_tr, y_age_val, y_age_te),
                "accent": (y_acc_tr, y_acc_val, y_acc_te),
            }

            for attr, (y_tr, y_val, y_te) in attributes.items():
                demo_acc[attr] = {
                    "demo_branch": {
                        "linear_val": linear_probe(
                            X_demo_tr, y_tr, X_demo_val, y_val
                        ),
                        "linear_test": linear_probe(
                            X_demo_tr, y_tr, X_demo_te, y_te
                        ),
                        "mlp_val": probe_with_ranges(
                            mlp_probe, X_demo_tr, y_tr, X_demo_val, y_val
                        ),
                        "mlp_test": probe_with_ranges(
                            mlp_probe, X_demo_tr, y_tr, X_demo_te, y_te
                        ),
                    },
                    "residual_branch": {
                        "linear_val": linear_probe(
                            X_task_tr, y_tr, X_task_val, y_val
                        ),
                        "linear_test": linear_probe(
                            X_task_tr, y_tr, X_task_te, y_te
                        ),
                        "mlp_val": probe_with_ranges(
                            mlp_probe, X_task_tr, y_tr, X_task_val, y_val
                        ),
                        "mlp_test": probe_with_ranges(
                            mlp_probe, X_task_tr, y_tr, X_task_te, y_te
                        ),
                    },
                }

            # --------------------------------------------------
            # Store results
            # --------------------------------------------------
            exp_result = {
                "k": k,
                "lambda_adv": lambdas,
                "speaker_verification": sv_metrics,
                "demo_acc": demo_acc,
            }
            results.append(exp_result)

            # --------------------------------------------------
            # Pretty print summary
            # --------------------------------------------------
            print("\nRESULT SUMMARY")
            print("-" * 80)
            print(f"Speaker verification:")
            print(f"  ROC-AUC : {sv_metrics['roc_auc']:.4f}")
            print(f"  EER     : {sv_metrics['eer']:.4f}")

            for attr in attributes:
                demo_lin = demo_acc[attr]["demo_branch"]["linear_test"]
                demo_mlp = demo_acc[attr]["demo_branch"]["mlp_test"]
                task_lin = demo_acc[attr]["residual_branch"]["linear_test"]
                task_mlp = demo_acc[attr]["residual_branch"]["mlp_test"]

                print(f"\n{attr.capitalize()} leakage:")
                print(
                    f"  Demo branch  - Linear : {demo_lin['acc']:.4f} "
                    f"(CI {demo_lin['ci_lower']:.4f}-{demo_lin['ci_upper']:.4f}), "
                    f"MLP : {demo_mlp['acc_mean']:.4f} ± {demo_mlp['acc_std']:.4f}"
                )
                print(
                    f"  Residual br. - Linear : {task_lin['acc']:.4f} "
                    f"(CI {task_lin['ci_lower']:.4f}-{task_lin['ci_upper']:.4f}), "
                    f"MLP : {task_mlp['acc_mean']:.4f} ± {task_mlp['acc_std']:.4f}"
                )

            print("-" * 80)

    return results
