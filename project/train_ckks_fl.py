import argparse
import copy
import csv
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.ckks_utils import (
    CKKSContextManager,
    CKKSUpdateEncryptor,
    EncryptedAggregator,
    TensorPacker,
    compute_model_update,
    apply_model_update,
    l2_norm_of_state_dict,
)
from src.fl_core import evaluate_model, train_one_client
from src.datasets import build_tiny_imagenet_datasets, dirichlet_partition_indices
from src.models import build_model
from src.utils import ensure_dir, save_json


def save_model_state(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    round_idx: int,
    val_loss: float,
    val_acc: float,
    best_val_acc: float,
) -> None:
    """
    Save a richer checkpoint so later phases can inspect round information.
    """
    ckpt = {
        "round": round_idx,
        "model_state_dict": model.state_dict(),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "best_val_acc_so_far": float(best_val_acc),
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    required_path_keys = [
        "output_root",
        "ckpt_dir",
        "log_dir",
        "metrics_dir",
        "plots_dir",
        "stats_dir",
    ]
    for key in required_path_keys:
        ensure_dir(config["paths"][key])

    # Extra folders for round-wise saving
    round_ckpt_dir = os.path.join(config["paths"]["ckpt_dir"], "per_round")
    round_stats_dir = os.path.join(config["paths"]["stats_dir"], "per_round")
    ensure_dir(round_ckpt_dir)
    ensure_dir(round_stats_dir)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    train_dataset, val_dataset = build_tiny_imagenet_datasets(
        config["paths"]["data_root"],
        config["dataset"]["image_size"],
    )

    client_indices = dirichlet_partition_indices(
        labels=np.array(train_dataset.targets),
        num_clients=config["fl"]["num_clients"],
        alpha=config["fl"]["dirichlet_alpha"],
        seed=config["seed"],
    )

    save_json(
        {str(i): list(map(int, idxs)) for i, idxs in enumerate(client_indices)},
        os.path.join(config["paths"]["stats_dir"], "client_indices.json"),
    )

    client_sample_counts = {str(i): int(len(idxs)) for i, idxs in enumerate(client_indices)}
    save_json(
        client_sample_counts,
        os.path.join(config["paths"]["stats_dir"], "client_sample_counts.json"),
    )

    client_loaders = []
    for idxs in client_indices:
        subset = Subset(train_dataset, idxs)
        loader = DataLoader(
            subset,
            batch_size=config["fl"]["batch_size"],
            shuffle=True,
            num_workers=config["dataloader"]["num_workers"],
            pin_memory=config["dataloader"]["pin_memory"],
        )
        client_loaders.append(loader)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["fl"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    global_model = build_model(
        model_name=config["model"]["name"],
        num_classes=config["dataset"]["num_classes"],
        pretrained=config["model"].get("pretrained", False),
    ).to(device)

    # ------------------------------------------------------------------
    # CKKS setup
    # ------------------------------------------------------------------
    ckks_manager = CKKSContextManager(
        poly_modulus_degree=config["ckks"]["poly_modulus_degree"],
        coeff_mod_bit_sizes=config["ckks"]["coeff_mod_bit_sizes"],
        global_scale_power=config["ckks"]["global_scale_power"],
    )
    encryptor = CKKSUpdateEncryptor(
        context=ckks_manager.context,
        chunk_size=config["ckks"]["chunk_size"],
    )
    aggregator = EncryptedAggregator()

    # ------------------------------------------------------------------
    # Output files
    # ------------------------------------------------------------------
    metrics_csv_path = os.path.join(config["paths"]["metrics_dir"], "ckks_round_metrics.csv")
    best_ckpt_path = os.path.join(config["paths"]["ckpt_dir"], "global_best_ckks.pt")
    last_ckpt_path = os.path.join(config["paths"]["ckpt_dir"], "global_last_ckks.pt")

    fieldnames = [
        "round",
        "avg_client_train_loss",
        "avg_client_train_acc",
        "global_val_loss",
        "global_val_acc",
        "avg_update_l2",
        "min_update_l2",
        "max_update_l2",
        "total_encrypt_time_sec",
        "aggregation_time_sec",
        "decrypt_time_sec",
        "total_ciphertext_bytes",
        "avg_ciphertext_bytes_per_client",
        "num_clients",
    ]

    best_val_acc = -1.0
    best_round = -1
    training_start_time = time.time()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for round_idx in range(1, config["fl"]["rounds"] + 1):
            print(f"\n===== Round {round_idx}/{config['fl']['rounds']} =====")

            round_start_time = time.time()
            global_state = copy.deepcopy(global_model.state_dict())

            encrypted_updates = []
            weights = []
            client_metrics = []
            update_norms = []

            total_encrypt_time = 0.0
            total_ciphertext_bytes = 0
            metadata = None

            # ----------------------------
            # Local client training
            # ----------------------------
            for client_id, loader in enumerate(tqdm(client_loaders, desc=f"Round {round_idx} Clients")):
                local_model = copy.deepcopy(global_model).to(device)

                local_state, metrics = train_one_client(
                    model=local_model,
                    dataloader=loader,
                    device=device,
                    epochs=config["fl"]["local_epochs"],
                    lr=config["fl"]["lr"],
                    momentum=config["fl"]["momentum"],
                    weight_decay=config["fl"]["weight_decay"],
                )

                update = compute_model_update(local_state, global_state)
                update_l2 = l2_norm_of_state_dict(update)
                update_norms.append(update_l2)

                flat_update, metadata = TensorPacker.state_dict_to_flat_vector(update)
                enc_update, stats = encryptor.encrypt_vector(flat_update)

                total_encrypt_time += stats["encrypt_time_sec"]
                total_ciphertext_bytes += stats["serialized_ciphertext_bytes"]

                encrypted_updates.append(enc_update)
                weights.append(metrics["num_samples"])
                client_metrics.append(
                    {
                        "client_id": client_id,
                        "train_loss": float(metrics["train_loss"]),
                        "train_acc": float(metrics["train_acc"]),
                        "num_samples": int(metrics["num_samples"]),
                        "update_l2": float(update_l2),
                        "encrypt_time_sec": float(stats["encrypt_time_sec"]),
                        "ciphertext_bytes": int(stats["serialized_ciphertext_bytes"]),
                    }
                )

            # ----------------------------
            # Encrypted aggregation
            # ----------------------------
            agg_start = time.time()
            enc_avg = aggregator.weighted_average(encrypted_updates, weights)
            aggregation_time = time.time() - agg_start

            # ----------------------------
            # Decryption
            # ----------------------------
            flat_avg, dec_stats = encryptor.decrypt_vector(enc_avg)
            decrypt_time = dec_stats["decrypt_time_sec"]

            avg_update = TensorPacker.flat_vector_to_state_dict(
                flat_vector=flat_avg,
                metadata=metadata,
                reference_state_dict=global_state,
                device=device,
            )

            new_state = apply_model_update(global_state, avg_update, device)
            global_model.load_state_dict(new_state)

            # ----------------------------
            # Validation
            # ----------------------------
            val_loss, val_acc = evaluate_model(global_model, val_loader, device)

            avg_train_loss = float(np.mean([m["train_loss"] for m in client_metrics]))
            avg_train_acc = float(np.mean([m["train_acc"] for m in client_metrics]))
            avg_update_l2 = float(np.mean(update_norms))
            min_update_l2 = float(np.min(update_norms))
            max_update_l2 = float(np.max(update_norms))
            avg_ciphertext_bytes_per_client = float(total_ciphertext_bytes / len(client_loaders))
            round_total_time = time.time() - round_start_time

            row = {
                "round": round_idx,
                "avg_client_train_loss": avg_train_loss,
                "avg_client_train_acc": avg_train_acc,
                "global_val_loss": float(val_loss),
                "global_val_acc": float(val_acc),
                "avg_update_l2": avg_update_l2,
                "min_update_l2": min_update_l2,
                "max_update_l2": max_update_l2,
                "total_encrypt_time_sec": float(total_encrypt_time),
                "aggregation_time_sec": float(aggregation_time),
                "decrypt_time_sec": float(decrypt_time),
                "total_ciphertext_bytes": int(total_ciphertext_bytes),
                "avg_ciphertext_bytes_per_client": avg_ciphertext_bytes_per_client,
                "num_clients": len(client_loaders),
            }

            writer.writerow(row)
            f_csv.flush()

            # ----------------------------
            # Save last checkpoint each round
            # ----------------------------
            save_checkpoint(
                model=global_model,
                path=last_ckpt_path,
                round_idx=round_idx,
                val_loss=val_loss,
                val_acc=val_acc,
                best_val_acc=max(best_val_acc, float(val_acc)),
            )

            # ----------------------------
            # Save per-round checkpoint
            # ----------------------------
            round_ckpt_path = os.path.join(round_ckpt_dir, f"global_round_{round_idx:03d}.pt")
            save_checkpoint(
                model=global_model,
                path=round_ckpt_path,
                round_idx=round_idx,
                val_loss=val_loss,
                val_acc=val_acc,
                best_val_acc=max(best_val_acc, float(val_acc)),
            )

            # ----------------------------
            # Save per-round JSON stats
            # ----------------------------
            round_stats = {
                "round": round_idx,
                "avg_client_train_loss": avg_train_loss,
                "avg_client_train_acc": avg_train_acc,
                "global_val_loss": float(val_loss),
                "global_val_acc": float(val_acc),
                "avg_update_l2": avg_update_l2,
                "min_update_l2": min_update_l2,
                "max_update_l2": max_update_l2,
                "total_encrypt_time_sec": float(total_encrypt_time),
                "aggregation_time_sec": float(aggregation_time),
                "decrypt_time_sec": float(decrypt_time),
                "total_ciphertext_bytes": int(total_ciphertext_bytes),
                "avg_ciphertext_bytes_per_client": avg_ciphertext_bytes_per_client,
                "num_clients": len(client_loaders),
                "round_total_time_sec": float(round_total_time),
                "clients": client_metrics,
            }
            save_json(
                round_stats,
                os.path.join(round_stats_dir, f"round_{round_idx:03d}.json"),
            )

            # ----------------------------
            # Save best checkpoint
            # ----------------------------
            if val_acc > best_val_acc:
                best_val_acc = float(val_acc)
                best_round = round_idx
                save_checkpoint(
                    model=global_model,
                    path=best_ckpt_path,
                    round_idx=round_idx,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    best_val_acc=best_val_acc,
                )

            print(
                f"Round {round_idx:03d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train Acc: {avg_train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )
            print(
                f"Encrypt: {total_encrypt_time:.2f}s | "
                f"Agg: {aggregation_time:.2f}s | "
                f"Decrypt: {decrypt_time:.2f}s | "
                f"Ciphertext: {total_ciphertext_bytes / (1024**3):.2f} GB"
            )

    total_training_time = time.time() - training_start_time

    # ------------------------------------------------------------------
    # Final run summary
    # ------------------------------------------------------------------
    run_summary = {
        "metrics_csv": metrics_csv_path,
        "best_checkpoint": best_ckpt_path,
        "last_checkpoint": last_ckpt_path,
        "per_round_checkpoint_dir": round_ckpt_dir,
        "per_round_stats_dir": round_stats_dir,
        "best_val_acc": float(best_val_acc),
        "best_round": int(best_round),
        "total_rounds": int(config["fl"]["rounds"]),
        "num_clients": int(config["fl"]["num_clients"]),
        "total_training_time_sec": float(total_training_time),
    }

    save_json(
        run_summary,
        os.path.join(config["paths"]["stats_dir"], "run_summary.json"),
    )

    print("\nCKKS training finished and all results were saved.")
    print(f"Best Val Acc: {best_val_acc:.2f}% at round {best_round}")
    print(f"Per-round checkpoints: {round_ckpt_dir}")
    print(f"Per-round stats: {round_stats_dir}")


if __name__ == "__main__":
    main()