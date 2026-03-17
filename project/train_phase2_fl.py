import argparse
import os
from copy import deepcopy

import torch

from src.datasets import get_tiny_imagenet_datasets, sanity_check_tiny_imagenet
from src.fl_client import FLClient
from src.fl_partition import (
    dirichlet_partition,
    compute_client_class_distribution,
    print_client_distribution,
    save_partition_stats,
)
from src.fl_server import fedavg_aggregate
from src.models import get_model
from src.utils import (
    load_yaml,
    set_seed,
    prepare_output_dirs,
    evaluate_model,
    save_checkpoint,
    save_metrics_csv,
    save_json,
    print_round_summary,
    get_timestamp,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Non-IID Federated Learning on Tiny-ImageNet-200")
    parser.add_argument("--config", type=str, default="config_phase2.yaml", help="Path to config YAML file")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)

    set_seed(config["seed"])
    prepare_output_dirs(config)

    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    print("=" * 80)
    print("Phase 2: Non-IID Federated Learning on Tiny-ImageNet-200")
    print("=" * 80)
    print(f"Using device: {device}")
    print(f"Config: {args.config}")

    # ---------------------------------------------------------------------
    # 1. Load Tiny-ImageNet-200
    # ---------------------------------------------------------------------
    train_dataset, val_dataset = get_tiny_imagenet_datasets(
        data_root=config["paths"]["data_root"],
        image_size=config["dataset"]["image_size"]
    )

    sanity_check_tiny_imagenet(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_classes=config["dataset"]["num_classes"]
    )

    # ---------------------------------------------------------------------
    # 2. Partition training data into non-IID client datasets (Dirichlet)
    # ---------------------------------------------------------------------
    client_to_indices = dirichlet_partition(
        targets=train_dataset.targets,
        num_clients=config["fl"]["num_clients"],
        num_classes=config["dataset"]["num_classes"],
        alpha=config["fl"]["dirichlet_alpha"],
        min_samples_per_client=config["fl"]["min_samples_per_client"],
        seed=config["seed"]
    )

    # ---------------------------------------------------------------------
    # 3. Print and save each client's class distribution
    # ---------------------------------------------------------------------
    distribution = compute_client_class_distribution(
        targets=train_dataset.targets,
        client_to_indices=client_to_indices,
        num_classes=config["dataset"]["num_classes"]
    )

    print_client_distribution(distribution)
    save_partition_stats(
        client_to_indices=client_to_indices,
        distribution=distribution,
        save_dir=config["paths"]["stats_dir"]
    )

    # Save full config used in this run
    save_json(config, os.path.join(config["paths"]["log_dir"], "used_config.json"))

    # ---------------------------------------------------------------------
    # 4. Build global model
    # ---------------------------------------------------------------------
    global_model = get_model(
        model_name=config["model"]["name"],
        num_classes=config["dataset"]["num_classes"],
        pretrained=config["model"]["pretrained"]
    )
    global_model.to(device)

    best_val_acc = -1.0
    round_metrics = []

    # Initial evaluation before FL starts
    initial_metrics = evaluate_model(
        model=global_model,
        dataset=val_dataset,
        batch_size=config["fl"]["eval_batch_size"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
        device=device
    )
    print(f"Initial Global Val Acc: {initial_metrics['val_acc']:.2f}%")

    # ---------------------------------------------------------------------
    # 5. Federated training loop across rounds
    # ---------------------------------------------------------------------
    for round_idx in range(1, config["fl"]["rounds"] + 1):
        print("\n" + "#" * 80)
        print(f"Starting Federated Round {round_idx}/{config['fl']['rounds']}")
        print("#" * 80)

        global_state = deepcopy(global_model.state_dict())
        client_results = []

        # 6. Local client training
        for client_id in range(config["fl"]["num_clients"]):
            client = FLClient(
                client_id=client_id,
                train_dataset=train_dataset,
                train_indices=client_to_indices[client_id],
                config=config,
                device=device
            )

            result = client.train(global_model_state=global_state)
            client_results.append(result)

            print(
                f"Client {client_id:02d} | samples={result['num_samples']} "
                f"| train_loss={result['train_loss']:.4f} | train_acc={result['train_acc']:.2f}%"
            )

        # 7. FedAvg aggregation
        aggregated_state = fedavg_aggregate(client_results)
        global_model.load_state_dict(aggregated_state)

        # 8. Validation after each round
        val_metrics = evaluate_model(
            model=global_model,
            dataset=val_dataset,
            batch_size=config["fl"]["eval_batch_size"],
            num_workers=config["dataloader"]["num_workers"],
            pin_memory=config["dataloader"]["pin_memory"],
            device=device
        )

        print_round_summary(round_idx, client_results, val_metrics)

        avg_client_loss = sum(r["train_loss"] for r in client_results) / len(client_results)
        avg_client_acc = sum(r["train_acc"] for r in client_results) / len(client_results)

        row = {
            "round": round_idx,
            "avg_client_train_loss": avg_client_loss,
            "avg_client_train_acc": avg_client_acc,
            "global_val_loss": val_metrics["val_loss"],
            "global_val_acc": val_metrics["val_acc"],
        }
        round_metrics.append(row)

        # 9. Save checkpoints and logs
        if config["save"]["save_every_round"]:
            ckpt_path = os.path.join(
                config["paths"]["ckpt_dir"],
                f"global_round_{round_idx}.pt"
            )
            save_checkpoint(
                {
                    "round": round_idx,
                    "model_state_dict": global_model.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config
                },
                ckpt_path
            )

        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            best_ckpt_path = os.path.join(config["paths"]["ckpt_dir"], "global_best.pt")
            save_checkpoint(
                {
                    "round": round_idx,
                    "model_state_dict": global_model.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config
                },
                best_ckpt_path
            )
            print(f"New best model saved with val_acc={best_val_acc:.2f}%")

    # 10. Save final model
    final_ckpt_path = os.path.join(config["paths"]["ckpt_dir"], "global_final.pt")
    save_checkpoint(
        {
            "round": config["fl"]["rounds"],
            "model_state_dict": global_model.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config
        },
        final_ckpt_path
    )

    # 11. Save metrics for paper
    metrics_csv_path = os.path.join(config["paths"]["metrics_dir"], "federated_round_metrics.csv")
    save_metrics_csv(round_metrics, metrics_csv_path)

    summary = {
        "best_val_acc": best_val_acc,
        "num_rounds": config["fl"]["rounds"],
        "num_clients": config["fl"]["num_clients"],
        "dirichlet_alpha": config["fl"]["dirichlet_alpha"],
        "timestamp": get_timestamp()
    }
    save_json(summary, os.path.join(config["paths"]["metrics_dir"], "summary.json"))

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final checkpoint: {final_ckpt_path}")
    print(f"Metrics CSV: {metrics_csv_path}")


if __name__ == "__main__":
    main()