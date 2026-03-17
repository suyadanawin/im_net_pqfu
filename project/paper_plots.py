import os
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Utility
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(output_path: str) -> None:
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 1) FL Convergence Curves
# Expected CSV columns:
# round, avg_client_train_loss, avg_client_train_acc, global_val_loss, global_val_acc
# ============================================================

def plot_fl_convergence(
    metrics_csv: str,
    output_dir: str
) -> None:
    df = load_csv(metrics_csv)
    ensure_dir(output_dir)

    # Global validation accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["global_val_acc"], marker="o")
    plt.xlabel("Federated Round")
    plt.ylabel("Global Validation Accuracy (%)")
    plt.title("FL Convergence: Global Validation Accuracy")
    plt.grid(True, alpha=0.3)
    save_fig(os.path.join(output_dir, "fl_global_val_accuracy.png"))

    # Global validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["global_val_loss"], marker="o")
    plt.xlabel("Federated Round")
    plt.ylabel("Global Validation Loss")
    plt.title("FL Convergence: Global Validation Loss")
    plt.grid(True, alpha=0.3)
    save_fig(os.path.join(output_dir, "fl_global_val_loss.png"))

    # Client train accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["avg_client_train_acc"], marker="o")
    plt.xlabel("Federated Round")
    plt.ylabel("Average Client Train Accuracy (%)")
    plt.title("FL Convergence: Average Client Train Accuracy")
    plt.grid(True, alpha=0.3)
    save_fig(os.path.join(output_dir, "fl_avg_client_train_accuracy.png"))

    # Client train loss
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["avg_client_train_loss"], marker="o")
    plt.xlabel("Federated Round")
    plt.ylabel("Average Client Train Loss")
    plt.title("FL Convergence: Average Client Train Loss")
    plt.grid(True, alpha=0.3)
    save_fig(os.path.join(output_dir, "fl_avg_client_train_loss.png"))

    # Combined train vs validation accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["avg_client_train_acc"], marker="o", label="Avg Client Train Acc")
    plt.plot(df["round"], df["global_val_acc"], marker="s", label="Global Val Acc")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy (%)")
    plt.title("FL Convergence: Train vs Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig(os.path.join(output_dir, "fl_train_vs_val_accuracy.png"))


# ============================================================
# 2) Method Comparison
# Expected CSV columns:
# method, accuracy, val_loss, latency_sec, weight_distance, mia_auc, communication_mb, storage_mb
# methods e.g. Full, Oracle, Unlearned
# ============================================================

def plot_method_comparison(
    comparison_csv: str,
    output_dir: str
) -> None:
    df = load_csv(comparison_csv)
    ensure_dir(output_dir)

    # Accuracy comparison
    plt.figure(figsize=(7, 5))
    plt.bar(df["method"], df["accuracy"])
    plt.ylabel("Accuracy (%)")
    plt.title("Method Comparison: Accuracy")
    plt.grid(axis="y", alpha=0.3)
    save_fig(os.path.join(output_dir, "method_accuracy_comparison.png"))

    # Validation loss comparison
    plt.figure(figsize=(7, 5))
    plt.bar(df["method"], df["val_loss"])
    plt.ylabel("Validation Loss")
    plt.title("Method Comparison: Validation Loss")
    plt.grid(axis="y", alpha=0.3)
    save_fig(os.path.join(output_dir, "method_val_loss_comparison.png"))

    # Unlearning latency
    if "latency_sec" in df.columns:
        plt.figure(figsize=(7, 5))
        plt.bar(df["method"], df["latency_sec"])
        plt.ylabel("Latency (sec)")
        plt.title("Method Comparison: Latency")
        plt.grid(axis="y", alpha=0.3)
        save_fig(os.path.join(output_dir, "method_latency_comparison.png"))

    # Weight distance
    if "weight_distance" in df.columns:
        plt.figure(figsize=(7, 5))
        plt.bar(df["method"], df["weight_distance"])
        plt.ylabel("L2 Distance to Oracle")
        plt.title("Method Comparison: Weight Distance")
        plt.grid(axis="y", alpha=0.3)
        save_fig(os.path.join(output_dir, "method_weight_distance_comparison.png"))

    # MIA AUC
    if "mia_auc" in df.columns:
        plt.figure(figsize=(7, 5))
        plt.bar(df["method"], df["mia_auc"])
        plt.ylabel("MIA ROC-AUC")
        plt.title("Method Comparison: MIA ROC-AUC")
        plt.grid(axis="y", alpha=0.3)
        save_fig(os.path.join(output_dir, "method_mia_auc_comparison.png"))

    # Communication overhead
    if "communication_mb" in df.columns:
        plt.figure(figsize=(7, 5))
        plt.bar(df["method"], df["communication_mb"])
        plt.ylabel("Communication Overhead (MB)")
        plt.title("Method Comparison: Communication Overhead")
        plt.grid(axis="y", alpha=0.3)
        save_fig(os.path.join(output_dir, "method_communication_comparison.png"))

    # Storage overhead
    if "storage_mb" in df.columns:
        plt.figure(figsize=(7, 5))
        plt.bar(df["method"], df["storage_mb"])
        plt.ylabel("Storage Overhead (MB)")
        plt.title("Method Comparison: Storage Overhead")
        plt.grid(axis="y", alpha=0.3)
        save_fig(os.path.join(output_dir, "method_storage_comparison.png"))


# ============================================================
# 3) Per-layer weight distance
# Expected CSV columns:
# layer, distance
# ============================================================

def plot_per_layer_distance(
    layer_distance_csv: str,
    output_dir: str
) -> None:
    df = load_csv(layer_distance_csv)
    ensure_dir(output_dir)

    plt.figure(figsize=(10, 5))
    plt.bar(df["layer"], df["distance"])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("L2 Distance")
    plt.title("Per-Layer Distance to Oracle")
    plt.grid(axis="y", alpha=0.3)
    save_fig(os.path.join(output_dir, "per_layer_weight_distance.png"))


# ============================================================
# 4) Client distribution heatmap
# Expected CSV format:
# rows = clients, columns = classes
# first column optionally 'client'
# ============================================================

def plot_client_distribution_heatmap(
    client_distribution_csv: str,
    output_dir: str
) -> None:
    df = load_csv(client_distribution_csv)
    ensure_dir(output_dir)

    if "client" in df.columns:
        client_labels = df["client"].astype(str).tolist()
        data = df.drop(columns=["client"]).values
    else:
        client_labels = [f"Client {i}" for i in range(len(df))]
        data = df.values

    plt.figure(figsize=(12, 6))
    plt.imshow(data, aspect="auto")
    plt.colorbar(label="Number of Samples")
    plt.yticks(np.arange(len(client_labels)), client_labels)
    plt.xlabel("Class Index")
    plt.ylabel("Client")
    plt.title("Client-Class Distribution Heatmap")
    save_fig(os.path.join(output_dir, "client_distribution_heatmap.png"))


# ============================================================
# 5) MIA confidence distribution
# Expected CSV columns:
# confidence, membership, method
# membership values: member / non-member
# ============================================================

def plot_mia_confidence_distribution(
    mia_conf_csv: str,
    output_dir: str
) -> None:
    df = load_csv(mia_conf_csv)
    ensure_dir(output_dir)

    methods = df["method"].unique().tolist()

    for method in methods:
        sub = df[df["method"] == method]
        member = sub[sub["membership"] == "member"]["confidence"].values
        non_member = sub[sub["membership"] == "non-member"]["confidence"].values

        plt.figure(figsize=(8, 5))
        plt.hist(member, bins=30, alpha=0.6, label="Member")
        plt.hist(non_member, bins=30, alpha=0.6, label="Non-member")
        plt.xlabel("Prediction Confidence")
        plt.ylabel("Frequency")
        plt.title(f"MIA Confidence Distribution: {method}")
        plt.legend()
        save_fig(os.path.join(output_dir, f"mia_confidence_distribution_{method.lower()}.png"))


# ============================================================
# 6) MIA ROC curves
# Expected CSV columns:
# method, fpr, tpr
# ============================================================

def plot_mia_roc_curves(
    mia_roc_csv: str,
    output_dir: str
) -> None:
    df = load_csv(mia_roc_csv)
    ensure_dir(output_dir)

    plt.figure(figsize=(7, 6))
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        plt.plot(sub["fpr"], sub["tpr"], label=method)

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MIA ROC Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig(os.path.join(output_dir, "mia_roc_curves.png"))


# ============================================================
# 7) CKKS noise vs utility
# Expected CSV columns:
# stage, noise_level, accuracy
# or
# noise_budget, accuracy
# ============================================================

def plot_ckks_noise_analysis(
    ckks_csv: str,
    output_dir: str
) -> None:
    df = load_csv(ckks_csv)
    ensure_dir(output_dir)

    if "noise_budget" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["noise_budget"], df["accuracy"], marker="o")
        plt.xlabel("Noise Budget / Proxy")
        plt.ylabel("Accuracy (%)")
        plt.title("CKKS Noise vs Accuracy")
        plt.grid(True, alpha=0.3)
        save_fig(os.path.join(output_dir, "ckks_noise_vs_accuracy.png"))

    if {"stage", "noise_level"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.bar(df["stage"], df["noise_level"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Noise Level / Proxy")
        plt.title("CKKS Noise Growth by Stage")
        plt.grid(axis="y", alpha=0.3)
        save_fig(os.path.join(output_dir, "ckks_noise_growth_by_stage.png"))

    if {"stage", "accuracy"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.bar(df["stage"], df["accuracy"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy by CKKS Processing Stage")
        plt.grid(axis="y", alpha=0.3)
        save_fig(os.path.join(output_dir, "ckks_accuracy_by_stage.png"))


# ============================================================
# 8) Ablation study
# Expected CSV columns:
# setting, accuracy, mia_auc, latency_sec
# ============================================================

def plot_ablation_study(
    ablation_csv: str,
    output_dir: str
) -> None:
    df = load_csv(ablation_csv)
    ensure_dir(output_dir)

    if "accuracy" in df.columns:
        plt.figure(figsize=(9, 5))
        plt.bar(df["setting"], df["accuracy"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Accuracy (%)")
        plt.title("Ablation Study: Accuracy")
        plt.grid(axis="y", alpha=0.3)
        save_fig(os.path.join(output_dir, "ablation_accuracy.png"))

    if "mia_auc" in df.columns:
        plt.figure(figsize=(9, 5))
        plt.bar(df["setting"], df["mia_auc"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("MIA ROC-AUC")
        plt.title("Ablation Study: MIA ROC-AUC")
        plt.grid(axis="y", alpha=0.3)
        save_fig(os.path.join(output_dir, "ablation_mia_auc.png"))

    if "latency_sec" in df.columns:
        plt.figure(figsize=(9, 5))
        plt.bar(df["setting"], df["latency_sec"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Latency (sec)")
        plt.title("Ablation Study: Latency")
        plt.grid(axis="y", alpha=0.3)
        save_fig(os.path.join(output_dir, "ablation_latency.png"))


# ============================================================
# 9) Forgetting-client effect across clients
# Expected CSV columns:
# forgotten_client, full_accuracy, oracle_accuracy, unlearned_accuracy, distance_to_oracle
# ============================================================

def plot_forgetting_client_effect(
    forgetting_csv: str,
    output_dir: str
) -> None:
    df = load_csv(forgetting_csv)
    ensure_dir(output_dir)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.25

    plt.bar(x - width, df["full_accuracy"], width=width, label="Full")
    plt.bar(x, df["oracle_accuracy"], width=width, label="Oracle")
    plt.bar(x + width, df["unlearned_accuracy"], width=width, label="Unlearned")

    plt.xticks(x, df["forgotten_client"].astype(str), rotation=30, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Forgotten Client")
    plt.title("Effect of Forgetting Different Clients")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    save_fig(os.path.join(output_dir, "forgetting_client_accuracy_comparison.png"))

    if "distance_to_oracle" in df.columns:
        plt.figure(figsize=(9, 5))
        plt.bar(df["forgotten_client"].astype(str), df["distance_to_oracle"])
        plt.xlabel("Forgotten Client")
        plt.ylabel("Distance to Oracle")
        plt.title("Distance to Oracle by Forgotten Client")
        plt.grid(axis="y", alpha=0.3)
        save_fig(os.path.join(output_dir, "forgetting_client_distance_to_oracle.png"))


# ============================================================
# 10) Main summary table exporter
# ============================================================

def export_summary_table(
    comparison_csv: str,
    output_dir: str
) -> None:
    df = load_csv(comparison_csv)
    ensure_dir(output_dir)

    summary_path = os.path.join(output_dir, "paper_summary_table.csv")
    df.to_csv(summary_path, index=False)
    print(f"Saved summary table to: {summary_path}")


# ============================================================
# 11) Example runner
# Edit file paths for your project
# ============================================================

def main() -> None:
    output_dir = "./paper_figures"
    ensure_dir(output_dir)

    # Phase 2 FL convergence
    fl_metrics_csv = "./outputs/metrics/fl_round_metrics.csv"
    if os.path.exists(fl_metrics_csv):
        plot_fl_convergence(fl_metrics_csv, output_dir)

    # Method comparison: Full / Oracle / Unlearned
    comparison_csv = "./outputs/metrics/method_comparison.csv"
    if os.path.exists(comparison_csv):
        plot_method_comparison(comparison_csv, output_dir)
        export_summary_table(comparison_csv, output_dir)

    # Per-layer distances
    layer_distance_csv = "./outputs/metrics/per_layer_distance.csv"
    if os.path.exists(layer_distance_csv):
        plot_per_layer_distance(layer_distance_csv, output_dir)

    # Client distribution
    client_distribution_csv = "./outputs/metrics/client_distribution.csv"
    if os.path.exists(client_distribution_csv):
        plot_client_distribution_heatmap(client_distribution_csv, output_dir)

    # MIA confidence distribution
    mia_conf_csv = "./outputs/metrics/mia_confidence.csv"
    if os.path.exists(mia_conf_csv):
        plot_mia_confidence_distribution(mia_conf_csv, output_dir)

    # MIA ROC curves
    mia_roc_csv = "./outputs/metrics/mia_roc.csv"
    if os.path.exists(mia_roc_csv):
        plot_mia_roc_curves(mia_roc_csv, output_dir)

    # CKKS noise analysis
    ckks_csv = "./outputs/metrics/ckks_noise_analysis.csv"
    if os.path.exists(ckks_csv):
        plot_ckks_noise_analysis(ckks_csv, output_dir)

    # Ablation study
    ablation_csv = "./outputs/metrics/ablation_study.csv"
    if os.path.exists(ablation_csv):
        plot_ablation_study(ablation_csv, output_dir)

    # Forgetting-client effect
    forgetting_csv = "./outputs/metrics/forgetting_client_effect.csv"
    if os.path.exists(forgetting_csv):
        plot_forgetting_client_effect(forgetting_csv, output_dir)

    print(f"All available paper plots saved to: {output_dir}")


if __name__ == "__main__":
    main()