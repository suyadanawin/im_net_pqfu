import argparse
import copy
import csv
import json
import os
import random
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_or_create_partitions(train_dataset, config):
    partition_file = config["paths"].get("partition_file", None)
    if partition_file and os.path.exists(partition_file):
        with open(partition_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        client_indices = [loaded[str(i)] for i in range(config["fl"]["num_clients"])]
        return client_indices

    client_indices = dirichlet_partition_indices(
        labels=np.array(train_dataset.targets),
        num_clients=config["fl"]["num_clients"],
        alpha=config["fl"]["dirichlet_alpha"],
        seed=config["seed"],
    )
    return client_indices


def save_model_state(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    forget_client_id = config["unlearning"]["forget_client_id"]

    oracle_root = config["paths"]["oracle_root"]
    ckpt_dir = os.path.join(oracle_root, "checkpoints")
    metrics_dir = os.path.join(oracle_root, "metrics")
    stats_dir = os.path.join(oracle_root, "stats")
    for p in [oracle_root, ckpt_dir, metrics_dir, stats_dir]:
        ensure_dir(p)

    train_dataset, val_dataset = build_tiny_imagenet_datasets(
        config["paths"]["data_root"],
        config["dataset"]["image_size"],
    )

    client_indices = load_or_create_partitions(train_dataset, config)

    retained_client_indices = [
        idxs for cid, idxs in enumerate(client_indices) if cid != forget_client_id
    ]
    retained_client_ids = [
        cid for cid in range(len(client_indices)) if cid != forget_client_id
    ]

    save_json(
        {
            "forget_client_id": int(forget_client_id),
            "retained_client_ids": retained_client_ids,
        },
        os.path.join(stats_dir, "oracle_setup.json"),
    )

    client_loaders = []
    for idxs in retained_client_indices:
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

    global_model = build_model(
        model_name=config["model"]["name"],
        num_classes=config["dataset"]["num_classes"],
        pretrained=config["model"].get("pretrained", False),
    ).to(device)

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

    metrics_csv = os.path.join(metrics_dir, "oracle_ckks_round_metrics.csv")
    best_ckpt = os.path.join(ckpt_dir, "oracle_best_ckks.pt")
    last_ckpt = os.path.join(ckpt_dir, "oracle_last_ckks.pt")

    fieldnames = [
        "round",
        "avg_client_train_loss",
        "avg_client_train_acc",
        "global_val_loss",
        "global_val_acc",
        "avg_update_l2",
        "total_encrypt_time_sec",
        "aggregation_time_sec",
        "decrypt_time_sec",
        "total_ciphertext_bytes",
        "num_retained_clients",
    ]

    best_val_acc = -1.0

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for round_idx in range(1, config["oracle"]["rounds"] + 1):
            print(f"\n===== Oracle CKKS Round {round_idx}/{config['oracle']['rounds']} =====")

            global_state = copy.deepcopy(global_model.state_dict())

            encrypted_updates = []
            weights = []
            client_metrics = []
            update_norms = []

            total_encrypt_time = 0.0
            total_ciphertext_bytes = 0
            metadata = None

            for client_id, loader in enumerate(tqdm(client_loaders, desc="Oracle clients")):
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
                update_norms.append(l2_norm_of_state_dict(update))

                flat_update, metadata = TensorPacker.state_dict_to_flat_vector(update)
                enc_update, stats = encryptor.encrypt_vector(flat_update)

                total_encrypt_time += stats["encrypt_time_sec"]
                total_ciphertext_bytes += stats["serialized_ciphertext_bytes"]

                encrypted_updates.append(enc_update)
                weights.append(metrics["num_samples"])
                client_metrics.append(metrics)

            start = time.time()
            enc_avg = aggregator.weighted_average(encrypted_updates, weights)
            agg_time = time.time() - start

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

            val_loss, val_acc = evaluate_model(global_model, val_loader, device)

            avg_train_loss = float(np.mean([m["train_loss"] for m in client_metrics]))
            avg_train_acc = float(np.mean([m["train_acc"] for m in client_metrics]))
            avg_update_l2 = float(np.mean(update_norms))

            row = {
                "round": round_idx,
                "avg_client_train_loss": avg_train_loss,
                "avg_client_train_acc": avg_train_acc,
                "global_val_loss": val_loss,
                "global_val_acc": val_acc,
                "avg_update_l2": avg_update_l2,
                "total_encrypt_time_sec": total_encrypt_time,
                "aggregation_time_sec": agg_time,
                "decrypt_time_sec": decrypt_time,
                "total_ciphertext_bytes": total_ciphertext_bytes,
                "num_retained_clients": len(client_loaders),
            }
            writer.writerow(row)
            f_csv.flush()

            save_model_state(global_model, last_ckpt)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model_state(global_model, best_ckpt)

            print(f"Oracle Val Acc: {val_acc:.2f}%")

    save_json(
        {
            "metrics_csv": metrics_csv,
            "best_checkpoint": best_ckpt,
            "last_checkpoint": last_ckpt,
            "best_val_acc": best_val_acc,
            "forget_client_id": forget_client_id,
        },
        os.path.join(stats_dir, "run_summary.json"),
    )

    print("\nOracle CKKS retraining finished.")


if __name__ == "__main__":
    main()