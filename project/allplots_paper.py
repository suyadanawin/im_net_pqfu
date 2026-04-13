import os
import sys
import argparse
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_curve, auc

sys.path.append(os.path.abspath("."))


# =========================
# Utilities
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_csv_if_exists(path: Optional[str]):
    if path is None or str(path).strip() == "":
        return None
    if not os.path.exists(path):
        print(f"[SKIP] File not found: {path}")
        return None
    return pd.read_csv(path)


def save_plot(output_path: str):
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {output_path}")


def find_column(df, candidates: List[str]):
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_first_existing_column(df, candidates: List[str]):
    return find_column(df, candidates)


def setup_paper_style():
    plt.rcParams.update({
        "figure.figsize": (8.6, 5.4),
        "figure.dpi": 120,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "lines.linewidth": 2.6,
        "lines.markersize": 7,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


def style_axis(ax):
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", width=1.0, length=5)


def annotate_bars(ax, bars, fmt="{:.2f}", offset_ratio=0.015):
    heights = [bar.get_height() for bar in bars]
    if not heights:
        return
    max_h = max(heights) if max(heights) != 0 else 1.0
    offset = max_h * offset_ratio
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )


def gb_formatter(x, pos):
    return f"{x:.3f}"


COLORS = {
    "train": "#1f77b4",
    "val": "#ff7f0e",
    "plain": "#1f77b4",
    "ckks": "#2ca02c",
    "full": "#1f77b4",
    "oracle": "#7f7f7f",
    "unlearn": "#d62728",
    "distance": "#9467bd",
    "gap": "#8c564b",
    "heatmap": "viridis",
}


def model_color(label: str):
    low = str(label).lower()
    if "full" in low:
        return COLORS["full"]
    if "oracle" in low:
        return COLORS["oracle"]
    if "unlearn" in low:
        return COLORS["unlearn"]
    if "plain" in low:
        return COLORS["plain"]
    if "ckks" in low:
        return COLORS["ckks"]
    return "#4c4c4c"


# =========================
# 1-4 Global / CKKS
# =========================
def plot_federated_training_accuracy_curve(phase2_df, out):
    round_col = find_column(phase2_df, ["round"])
    train_col = find_column(phase2_df, ["avg_client_train_acc", "train_acc"])
    val_col = find_column(phase2_df, ["global_val_acc", "val_acc"])
    if None in [round_col, train_col, val_col]:
        return
    fig, ax = plt.subplots()
    ax.plot(phase2_df[round_col], phase2_df[train_col], marker="o", color=COLORS["train"], label="Train Accuracy")
    ax.plot(phase2_df[round_col], phase2_df[val_col], marker="s", color=COLORS["val"], label="Validation Accuracy")
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Federated Training Accuracy Curve")
    ax.legend(loc="upper left")
    style_axis(ax)
    save_plot(os.path.join(out, "fig_federated_training_accuracy_curve.png"))


def plot_federated_training_loss_curve(phase2_df, out):
    round_col = find_column(phase2_df, ["round"])
    train_col = find_column(phase2_df, ["avg_client_train_loss", "train_loss"])
    val_col = find_column(phase2_df, ["global_val_loss", "val_loss"])
    if None in [round_col, train_col, val_col]:
        return
    fig, ax = plt.subplots()
    ax.plot(phase2_df[round_col], phase2_df[train_col], marker="o", color=COLORS["train"], label="Train Loss")
    ax.plot(phase2_df[round_col], phase2_df[val_col], marker="s", color=COLORS["val"], label="Validation Loss")
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.set_title("Federated Training Loss Curve")
    ax.legend(loc="upper right")
    style_axis(ax)
    save_plot(os.path.join(out, "fig_federated_training_loss_curve.png"))


def plot_global_validation_accuracy_per_round(phase2_df, out):
    round_col = find_column(phase2_df, ["round"])
    val_col = find_column(phase2_df, ["global_val_acc", "val_acc"])
    if None in [round_col, val_col]:
        return
    fig, ax = plt.subplots()
    ax.plot(phase2_df[round_col], phase2_df[val_col], marker="o", color=COLORS["val"])
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Global Validation Accuracy per Round")
    style_axis(ax)
    save_plot(os.path.join(out, "fig_global_validation_accuracy_per_round.png"))


def plot_global_utility_comparison(phase2_df, ckks_df, out):
    plain_round = find_column(phase2_df, ["round"])
    plain_val = find_column(phase2_df, ["global_val_acc", "val_acc"])
    ckks_round = find_column(ckks_df, ["round"])
    ckks_val = find_column(ckks_df, ["global_val_acc", "val_acc"])

    if None in [plain_round, plain_val, ckks_round, ckks_val]:
        print("[SKIP] Missing required columns for global utility comparison.")
        return

    # Make safe copies
    phase2_df = phase2_df.copy()
    ckks_df = ckks_df.copy()

    # Force numeric conversion
    phase2_df[plain_round] = pd.to_numeric(phase2_df[plain_round], errors="coerce")
    phase2_df[plain_val] = pd.to_numeric(phase2_df[plain_val], errors="coerce")

    ckks_df[ckks_round] = pd.to_numeric(ckks_df[ckks_round], errors="coerce")
    ckks_df[ckks_val] = pd.to_numeric(ckks_df[ckks_val], errors="coerce")

    # Drop invalid rows
    phase2_df = phase2_df.dropna(subset=[plain_round, plain_val])
    ckks_df = ckks_df.dropna(subset=[ckks_round, ckks_val])

    if phase2_df.empty:
        print("[SKIP] Phase2 dataframe is empty after cleaning.")
        return

    if ckks_df.empty:
        print("[SKIP] CKKS dataframe is empty after cleaning.")
        return

    max_ckks_round = int(ckks_df[ckks_round].max())
    plain_subset = phase2_df[phase2_df[plain_round] <= max_ckks_round].copy()

    if plain_subset.empty:
        print("[SKIP] No matching plaintext rounds up to CKKS max round.")
        return

    fig, ax = plt.subplots()
    ax.plot(
        plain_subset[plain_round],
        plain_subset[plain_val],
        marker="o",
        color=COLORS["plain"],
        label="Plain FL"
    )
    ax.plot(
        ckks_df[ckks_round],
        ckks_df[ckks_val],
        marker="s",
        color=COLORS["ckks"],
        label="CKKS FL"
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Global Utility Comparison")
    ax.legend(loc="upper left")
    style_axis(ax)
    save_plot(os.path.join(out, "fig1_global_utility_comparison.png"))

def plot_training_time_per_round(ckks_df, out):
    round_col = find_column(ckks_df, ["round"])
    enc = find_column(ckks_df, ["total_encrypt_time_sec"])
    agg = find_column(ckks_df, ["aggregation_time_sec"])
    dec = find_column(ckks_df, ["decrypt_time_sec"])
    if None in [round_col, enc, agg, dec]:
        return
    total = ckks_df[enc] + ckks_df[agg] + ckks_df[dec]
    fig, ax = plt.subplots()
    ax.plot(ckks_df[round_col], total, marker="o", color=COLORS["ckks"])
    ax.set_xlabel("Round")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Training Time per Round")
    style_axis(ax)
    save_plot(os.path.join(out, "fig_training_time_per_round.png"))


def plot_encryption_overhead(ckks_df, out):
    enc = find_column(ckks_df, ["total_encrypt_time_sec"])
    agg = find_column(ckks_df, ["aggregation_time_sec"])
    dec = find_column(ckks_df, ["decrypt_time_sec"])
    if None in [enc, agg, dec]:
        return

    labels = ["Encryption", "Aggregation", "Decryption"]
    values = [ckks_df[enc].mean(), ckks_df[agg].mean(), ckks_df[dec].mean()]
    colors = [COLORS["ckks"], "#17becf", "#bcbd22"]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Encryption Overhead")
    style_axis(ax)
    annotate_bars(ax, bars)
    save_plot(os.path.join(out, "fig_encryption_overhead.png"))


def plot_communication_cost_per_round(ckks_df, out):
    round_col = find_column(ckks_df, ["round"])
    size_col = find_column(ckks_df, ["total_ciphertext_bytes"])

    if None in [round_col, size_col]:
        print("[SKIP] Missing required columns for communication cost plot.")
        return

    df = ckks_df.copy()
    df[round_col] = pd.to_numeric(df[round_col], errors="coerce")
    df[size_col] = pd.to_numeric(df[size_col], errors="coerce")
    df = df.dropna(subset=[round_col, size_col])

    if df.empty:
        print("[SKIP] CKKS communication dataframe is empty after cleaning.")
        return

    gb = df[size_col] / 1e9
    gb = gb.replace([np.inf, -np.inf], np.nan)
    valid = gb.notna()

    if valid.sum() == 0:
        print("[SKIP] No valid communication cost values available.")
        return

    df = df.loc[valid].copy()
    gb = gb.loc[valid]

    fig, ax = plt.subplots()
    ax.plot(df[round_col], gb, marker="o", color=COLORS["ckks"])
    ax.set_xlabel("Round")
    ax.set_ylabel("Communication Cost (GB)")
    ax.set_title("Communication Cost per Round")
    ax.yaxis.set_major_formatter(FuncFormatter(gb_formatter))

    y_min = float(gb.min())
    y_max = float(gb.max())

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        print("[SKIP] Communication cost axis limits are not finite.")
        plt.close()
        return

    if y_min == y_max:
        margin = max(abs(y_min) * 0.15, 0.0001)
    else:
        margin = max((y_max - y_min) * 0.15, 0.0001)

    ax.set_ylim(y_min - margin, y_max + margin)
    style_axis(ax)
    save_plot(os.path.join(out, "fig_communication_cost_per_round.png"))


def plot_model_size_overhead_comparison(phase2_df, ckks_df, out):
    size_col = find_column(ckks_df, ["total_ciphertext_bytes"])
    if size_col is None:
        return
    labels = ["Plain", "CKKS"]
    values = [0.0, float(ckks_df[size_col].mean() / 1e9)]
    colors = [COLORS["plain"], COLORS["ckks"]]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Average Communication (GB)")
    ax.set_title("Model Size / Communication Overhead Comparison")
    style_axis(ax)
    annotate_bars(ax, bars)
    save_plot(os.path.join(out, "fig_model_size_overhead_comparison.png"))


def plot_ckks_noise_growth_curve(ckks_df, out):
    # placeholder proxy if explicit noise is unavailable
    round_col = find_column(ckks_df, ["round"])
    size_col = find_column(ckks_df, ["total_ciphertext_bytes"])
    if None in [round_col, size_col]:
        return
    proxy = ckks_df[size_col] / ckks_df[size_col].iloc[0]
    fig, ax = plt.subplots()
    ax.plot(ckks_df[round_col], proxy, marker="o", color=COLORS["distance"])
    ax.set_xlabel("Round")
    ax.set_ylabel("Relative Noise/Scale Proxy")
    ax.set_title("CKKS Noise Growth Curve")
    style_axis(ax)
    save_plot(os.path.join(out, "fig_ckks_noise_growth_curve.png"))


def plot_accuracy_vs_ckks_noise_level(phase2_df, ckks_df, out):
    ckks_round = find_column(ckks_df, ["round"])
    ckks_val = find_column(ckks_df, ["global_val_acc", "val_acc"])
    size_col = find_column(ckks_df, ["total_ciphertext_bytes"])
    if None in [ckks_round, ckks_val, size_col]:
        return
    proxy = ckks_df[size_col] / 1e9
    fig, ax = plt.subplots()
    ax.scatter(proxy, ckks_df[ckks_val], s=80, color=COLORS["ckks"])
    ax.set_xlabel("Ciphertext Size Proxy (GB)")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Accuracy vs CKKS Noise Level")
    style_axis(ax)
    save_plot(os.path.join(out, "fig_accuracy_vs_ckks_noise_level.png"))


# =========================
# 5-17 Client / Unlearning / Privacy
# =========================
def plot_model_accuracy_comparison(df, client_name, out):
    model_col = get_first_existing_column(df, ["Model", "model", "Comparison", "comparison"])
    val_col = get_first_existing_column(df, ["Val Acc (%)", "val_acc", "Validation Accuracy", "val_acc_percent"])
    if None in [model_col, val_col]:
        return
    labels = df[model_col].astype(str).tolist()
    colors = [model_color(x) for x in labels]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, df[val_col], color=colors)
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(f"Model Comparison ({client_name})")
    style_axis(ax)
    annotate_bars(ax, bars)
    save_plot(os.path.join(out, f"fig_model_compare_{client_name}.png"))


def plot_mia_score_comparison(df, client_name, out):
    model_col = get_first_existing_column(df, ["Model", "model", "Comparison", "comparison"])
    mia_col = get_first_existing_column(df, ["MIA", "mia", "MIA Score", "mia_score", "mia_acc"])
    if None in [model_col, mia_col]:
        return
    labels = df[model_col].astype(str).tolist()
    colors = [model_color(x) for x in labels]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, df[mia_col], color=colors)
    ax.axhline(0.5, linestyle="--", linewidth=1.4, color="black", alpha=0.7, label="Random Guess")
    ax.set_ylabel("MIA Score")
    ax.set_title(f"MIA Comparison ({client_name})")
    ax.legend(loc="upper right")
    style_axis(ax)
    annotate_bars(ax, bars, fmt="{:.3f}", offset_ratio=0.02)
    save_plot(os.path.join(out, f"fig_mia_{client_name}.png"))


def plot_gap(df, client_name, out):
    round_col = find_column(df, ["round"])
    train_col = find_column(df, ["train_acc", "avg_client_train_acc"])
    val_col = find_column(df, ["val_acc", "global_val_acc"])
    if None in [round_col, train_col, val_col]:
        return
    gap = df[train_col] - df[val_col]
    fig, ax = plt.subplots()
    ax.plot(df[round_col], gap, marker="o", color=COLORS["gap"])
    ax.set_xlabel("Correction Round")
    ax.set_ylabel("Train–Validation Gap")
    ax.set_title(f"Gap Analysis ({client_name})")
    style_axis(ax)
    save_plot(os.path.join(out, f"fig_gap_{client_name}.png"))


def plot_fidelity(distance_df, client_name, out):
    label_col = get_first_existing_column(distance_df, ["Model", "model", "Comparison", "comparison", "pair"])
    full_col = get_first_existing_column(distance_df, ["Distance to Oracle", "distance_to_oracle", "Oracle Distance", "full_model_l2"])
    if full_col is None:
        return
    labels = distance_df[label_col].astype(str).tolist() if label_col else [str(i) for i in range(len(distance_df))]
    colors = [model_color(x) for x in labels]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, distance_df[full_col], color=colors)
    ax.set_ylabel("Distance to Oracle")
    ax.set_title(f"Fidelity ({client_name})")
    style_axis(ax)
    annotate_bars(ax, bars)
    save_plot(os.path.join(out, f"fig_fidelity_{client_name}.png"))


def plot_heatmap(unlearn_per_layer_df, client_name, out):
    round_col = find_column(unlearn_per_layer_df, ["round"])
    if round_col is None:
        return
    numeric_cols = [c for c in unlearn_per_layer_df.columns if c != round_col and pd.api.types.is_numeric_dtype(unlearn_per_layer_df[c])]
    if not numeric_cols:
        return
    heat = unlearn_per_layer_df[numeric_cols].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heat.T.values, aspect="auto", cmap=COLORS["heatmap"])
    ax.set_xlabel("Round Index")
    ax.set_ylabel("Layer / Metric")
    ax.set_title(f"Layer-wise Heatmap ({client_name})")
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols)
    plt.colorbar(im, ax=ax)
    save_plot(os.path.join(out, f"fig_heatmap_{client_name}.png"))


def plot_unlearning_validation_accuracy(df, client_name, out):
    round_col = find_column(df, ["round"])
    val_col = find_column(df, ["val_acc", "global_val_acc"])
    if None in [round_col, val_col]:
        return
    fig, ax = plt.subplots()
    ax.plot(df[round_col], df[val_col], marker="s", color=COLORS["unlearn"])
    ax.set_xlabel("Correction Round")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(f"Unlearning Validation Accuracy ({client_name})")
    style_axis(ax)
    save_plot(os.path.join(out, f"fig_unlearning_validation_accuracy_{client_name}.png"))


def plot_unlearning_convergence_curve(df, client_name, out):
    round_col = find_column(df, ["round"])
    train_col = find_column(df, ["train_acc", "avg_client_train_acc"])
    val_col = find_column(df, ["val_acc", "global_val_acc"])
    if None in [round_col, train_col, val_col]:
        return
    fig, ax = plt.subplots()
    ax.plot(df[round_col], df[train_col], marker="o", color=COLORS["train"], label="Train Accuracy")
    ax.plot(df[round_col], df[val_col], marker="s", color=COLORS["unlearn"], label="Validation Accuracy")
    ax.set_xlabel("Correction Round")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Unlearning Convergence ({client_name})")
    ax.legend(loc="upper left")
    style_axis(ax)
    save_plot(os.path.join(out, f"fig_unlearning_convergence_{client_name}.png"))


def plot_full_model_vs_oracle_l2_distance(distance_df, client_name, out):
    pair_col = get_first_existing_column(distance_df, ["pair", "Comparison", "comparison"])
    full_col = get_first_existing_column(distance_df, ["full_model_l2", "Distance to Oracle", "distance_to_oracle"])
    if None in [pair_col, full_col]:
        return
    fig, ax = plt.subplots()
    bars = ax.bar(distance_df[pair_col].astype(str), distance_df[full_col], color=[model_color(x) for x in distance_df[pair_col]])
    ax.set_ylabel("Full Model L2 Distance")
    ax.set_title(f"Full Model Distance to Oracle ({client_name})")
    ax.tick_params(axis="x", rotation=20)
    style_axis(ax)
    annotate_bars(ax, bars)
    save_plot(os.path.join(out, f"fig_full_model_distance_{client_name}.png"))


def plot_last_layer_distance_to_oracle(distance_df, client_name, out):
    pair_col = get_first_existing_column(distance_df, ["pair", "Comparison", "comparison"])
    last_col = get_first_existing_column(distance_df, ["last_layer_l2", "Last Layer L2"])
    if None in [pair_col, last_col]:
        return
    fig, ax = plt.subplots()
    bars = ax.bar(distance_df[pair_col].astype(str), distance_df[last_col], color=[model_color(x) for x in distance_df[pair_col]])
    ax.set_ylabel("Last Layer L2 Distance")
    ax.set_title(f"Last-Layer Distance to Oracle ({client_name})")
    ax.tick_params(axis="x", rotation=20)
    style_axis(ax)
    annotate_bars(ax, bars)
    save_plot(os.path.join(out, f"fig_last_layer_distance_{client_name}.png"))


def plot_distance_to_oracle_vs_unlearning_rounds(oracle_df, unlearn_df, client_name, out):
    r1 = find_column(oracle_df, ["round"])
    r2 = find_column(unlearn_df, ["round"])
    d1 = get_first_existing_column(oracle_df, ["full_model_l2", "distance_to_oracle", "Oracle Distance", "last_layer_l2"])
    d2 = get_first_existing_column(unlearn_df, ["full_model_l2", "distance_to_oracle", "Oracle Distance", "last_layer_l2"])
    if None in [r1, r2, d1, d2]:
        return
    fig, ax = plt.subplots()
    ax.plot(oracle_df[r1], oracle_df[d1], marker="o", color=COLORS["oracle"], label="Oracle Roundwise")
    ax.plot(unlearn_df[r2], unlearn_df[d2], marker="s", color=COLORS["unlearn"], label="Unlearn Roundwise")
    ax.set_xlabel("Unlearning Round")
    ax.set_ylabel("Distance to Oracle")
    ax.set_title(f"Distance to Oracle vs Unlearning Rounds ({client_name})")
    ax.legend(loc="upper right")
    style_axis(ax)
    save_plot(os.path.join(out, f"fig_distance_vs_rounds_{client_name}.png"))


def plot_layer_wise_distance_to_oracle(unlearn_per_layer_df, client_name, out):
    round_col = find_column(unlearn_per_layer_df, ["round"])
    if round_col is None:
        return
    numeric_cols = [c for c in unlearn_per_layer_df.columns if c != round_col and pd.api.types.is_numeric_dtype(unlearn_per_layer_df[c])]
    if not numeric_cols:
        return
    means = unlearn_per_layer_df[numeric_cols].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(means.index.astype(str), means.values, color=COLORS["distance"])
    ax.set_ylabel("Average L2 Distance")
    ax.set_title(f"Layer-wise Distance to Oracle ({client_name})")
    ax.tick_params(axis="x", rotation=65)
    style_axis(ax)
    save_plot(os.path.join(out, f"fig_layerwise_distance_{client_name}.png"))


def plot_parameter_update_norm_per_round(df, client_name, out):
    round_col = find_column(df, ["round"])
    train_col = find_column(df, ["train_acc", "avg_client_train_acc"])
    val_col = find_column(df, ["val_acc", "global_val_acc"])
    if None in [round_col, train_col, val_col]:
        return
    proxy = (df[train_col] - df[val_col]).abs()
    fig, ax = plt.subplots()
    ax.plot(df[round_col], proxy, marker="o", color=COLORS["distance"])
    ax.set_xlabel("Round")
    ax.set_ylabel("Update Norm Proxy")
    ax.set_title(f"Parameter Update Norm per Round ({client_name})")
    style_axis(ax)
    save_plot(os.path.join(out, f"fig_update_norm_{client_name}.png"))


def plot_utility_retention_vs_oracle_gap(compare_df, client_name, out):
    model_col = get_first_existing_column(compare_df, ["Model", "model", "Comparison", "comparison"])
    val_col = get_first_existing_column(compare_df, ["Val Acc (%)", "val_acc", "Validation Accuracy", "val_acc_percent"])
    if None in [model_col, val_col]:
        return
    oracle_rows = compare_df[compare_df[model_col].astype(str).str.lower().str.contains("oracle")]
    if oracle_rows.empty:
        return
    oracle_val = float(oracle_rows[val_col].iloc[0])
    gap = compare_df[val_col] - oracle_val
    labels = compare_df[model_col].astype(str).tolist()
    colors = [model_color(x) for x in labels]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, gap, color=colors)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_ylabel("Accuracy Gap vs Oracle (%)")
    ax.set_title(f"Utility Retention vs Oracle Gap ({client_name})")
    style_axis(ax)
    annotate_bars(ax, bars)
    save_plot(os.path.join(out, f"fig_utility_retention_vs_oracle_gap_{client_name}.png"))


def plot_accuracy_vs_privacy_tradeoff(compare_df, mia_df, client_name, out):
    model_col_cmp = get_first_existing_column(compare_df, ["Model", "model", "Comparison", "comparison"])
    val_col = get_first_existing_column(compare_df, ["Val Acc (%)", "val_acc", "Validation Accuracy", "val_acc_percent"])
    model_col_mia = get_first_existing_column(mia_df, ["Model", "model", "Comparison", "comparison"])
    mia_col = get_first_existing_column(mia_df, ["MIA", "mia", "MIA Score", "mia_score", "mia_acc"])
    if None in [model_col_cmp, val_col, model_col_mia, mia_col]:
        return
    merged = pd.merge(
        compare_df[[model_col_cmp, val_col]].rename(columns={model_col_cmp: "model"}),
        mia_df[[model_col_mia, mia_col]].rename(columns={model_col_mia: "model"}),
        on="model",
        how="inner"
    )
    if merged.empty:
        return
    fig, ax = plt.subplots()
    for _, row in merged.iterrows():
        color = model_color(row["model"])
        ax.scatter(row[val_col], row[mia_col], s=90, color=color)
        ax.text(row[val_col] + 0.08, row[mia_col] + 0.002, str(row["model"]), fontsize=10, weight="bold")
    ax.set_xlabel("Validation Accuracy (%)")
    ax.set_ylabel("MIA Score")
    ax.set_title(f"Accuracy vs Privacy Trade-off ({client_name})")
    style_axis(ax)
    save_plot(os.path.join(out, f"fig_tradeoff_{client_name}.png"))


def plot_mia_roc_curve(mia_df, client_name, out):
    # approximate ROC from means if raw scores unavailable
    required = ["member_mean_conf", "nonmember_mean_conf"]
    if not all(c in mia_df.columns for c in required):
        return
    models = mia_df["model"].astype(str).tolist()
    fig, ax = plt.subplots()
    for _, row in mia_df.iterrows():
        member = float(row["member_mean_conf"])
        nonmember = float(row["nonmember_mean_conf"])
        y_true = np.array([1] * 50 + [0] * 50)
        y_scores = np.array([member] * 50 + [nonmember] * 50)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{row['model']} (AUC={roc_auc:.3f})", color=model_color(row["model"]))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"MIA ROC Curve ({client_name})")
    ax.legend(loc="lower right")
    style_axis(ax)
    save_plot(os.path.join(out, f"fig_mia_roc_{client_name}.png"))


def plot_mia_auc_comparison(mia_df, client_name, out):
    if not all(c in mia_df.columns for c in ["model", "member_mean_conf", "nonmember_mean_conf"]):
        return
    auc_rows = []
    for _, row in mia_df.iterrows():
        y_true = np.array([1] * 50 + [0] * 50)
        y_scores = np.array([float(row["member_mean_conf"])] * 50 + [float(row["nonmember_mean_conf"])] * 50)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_rows.append((str(row["model"]), auc(fpr, tpr)))
    labels = [x[0] for x in auc_rows]
    values = [x[1] for x in auc_rows]
    colors = [model_color(x) for x in labels]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("ROC AUC")
    ax.set_title(f"MIA AUC Comparison ({client_name})")
    style_axis(ax)
    annotate_bars(ax, bars, fmt="{:.3f}")
    save_plot(os.path.join(out, f"fig_mia_auc_{client_name}.png"))


# =========================
# 18-29 Multi-client / optional
# =========================
def plot_multi_client_accuracy_comparison(compare_dfs, out):
    rows = []
    for client_name, df in compare_dfs.items():
        if df is None:
            continue
        model_col = get_first_existing_column(df, ["Model", "model", "Comparison", "comparison"])
        val_col = get_first_existing_column(df, ["Val Acc (%)", "val_acc", "Validation Accuracy", "val_acc_percent"])
        if None in [model_col, val_col]:
            continue
        for _, row in df.iterrows():
            rows.append({"client": client_name, "model": str(row[model_col]), "val": float(row[val_col])})
    if not rows:
        return
    merged = pd.DataFrame(rows)
    fig, ax = plt.subplots()
    for model in merged["model"].unique():
        sub = merged[merged["model"] == model].copy()
        ax.plot(sub["client"], sub["val"], marker="o", label=model, color=model_color(model))
    ax.set_xlabel("Client")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Multi-Client Accuracy Comparison")
    ax.legend(loc="best")
    style_axis(ax)
    save_plot(os.path.join(out, "fig_multi_client_accuracy_comparison.png"))


def plot_per_client_accuracy_distribution(compare_dfs, out):
    rows = []
    for client_name, df in compare_dfs.items():
        if df is None:
            continue
        model_col = get_first_existing_column(df, ["Model", "model", "Comparison", "comparison"])
        val_col = get_first_existing_column(df, ["Val Acc (%)", "val_acc", "Validation Accuracy", "val_acc_percent"])
        if None in [model_col, val_col]:
            continue
        un_rows = df[df[model_col].astype(str).str.lower().str.contains("unlearn")]
        if un_rows.empty:
            continue
        rows.append((client_name, float(un_rows[val_col].iloc[0])))
    if not rows:
        return
    labels = [x[0] for x in rows]
    values = [x[1] for x in rows]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=COLORS["unlearn"])
    ax.set_xlabel("Client")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Per-Client Accuracy Distribution")
    style_axis(ax)
    annotate_bars(ax, bars)
    save_plot(os.path.join(out, "fig_per_client_accuracy_distribution.png"))


def plot_client_data_distribution_heatmap(indices_json_path, out):
    if not os.path.exists(indices_json_path):
        return
    import json
    with open(indices_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    clients = sorted(data.keys(), key=lambda x: int(str(x).replace("client_", "").replace("client", "").strip("_")) if str(x).replace("client_", "").replace("client", "").strip("_").isdigit() else str(x))
    counts = np.array([[len(data[c])] for c in clients], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(counts, aspect="auto", cmap=COLORS["heatmap"])
    ax.set_xlabel("Sample Count")
    ax.set_ylabel("Client")
    ax.set_title("Client Data Distribution Heatmap")
    ax.set_yticks(range(len(clients)))
    ax.set_yticklabels(clients)
    plt.colorbar(im, ax=ax)
    save_plot(os.path.join(out, "fig_client_data_distribution_heatmap.png"))


def plot_class_distribution_per_client(indices_json_path, out):
    # placeholder if only client sizes are available
    if not os.path.exists(indices_json_path):
        return
    import json
    with open(indices_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = list(data.keys())
    sizes = [len(v) for v in data.values()]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, sizes, color=COLORS["plain"])
    ax.set_xlabel("Client")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution per Client")
    ax.tick_params(axis="x", rotation=20)
    style_axis(ax)
    annotate_bars(ax, bars, fmt="{:.0f}")
    save_plot(os.path.join(out, "fig_class_distribution_per_client.png"))


def plot_accuracy_vs_number_of_clients(compare_dfs, out):
    rows = []
    for client_name, df in compare_dfs.items():
        if df is None:
            continue
        model_col = get_first_existing_column(df, ["Model", "model", "Comparison", "comparison"])
        val_col = get_first_existing_column(df, ["Val Acc (%)", "val_acc", "Validation Accuracy", "val_acc_percent"])
        if None in [model_col, val_col]:
            continue
        full_rows = df[df[model_col].astype(str).str.lower().str.contains("full")]
        if full_rows.empty:
            continue
        rows.append(float(full_rows[val_col].iloc[0]))
    if not rows:
        return
    x = np.arange(1, len(rows) + 1)
    fig, ax = plt.subplots()
    ax.plot(x, rows, marker="o", color=COLORS["plain"])
    ax.set_xlabel("Client Index Count Proxy")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Accuracy vs Number of Clients")
    style_axis(ax)
    save_plot(os.path.join(out, "fig_accuracy_vs_number_of_clients.png"))


def plot_accuracy_vs_dirichlet_alpha(out):
    # placeholder static plot for future experiments
    alphas = [0.1]
    accs = [np.nan]
    fig, ax = plt.subplots()
    ax.plot(alphas, accs, marker="o", color=COLORS["plain"])
    ax.set_xlabel("Dirichlet Alpha")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Accuracy vs Dirichlet Alpha")
    style_axis(ax)
    save_plot(os.path.join(out, "fig_accuracy_vs_dirichlet_alpha.png"))


def plot_combined_performance_summary(compare_df, mia_df, distance_df, client_name, out):
    if compare_df is None or mia_df is None or distance_df is None:
        return
    model_col = get_first_existing_column(compare_df, ["Model", "model", "Comparison", "comparison"])
    val_col = get_first_existing_column(compare_df, ["Val Acc (%)", "val_acc", "Validation Accuracy", "val_acc_percent"])
    mia_model = get_first_existing_column(mia_df, ["Model", "model", "Comparison", "comparison"])
    mia_col = get_first_existing_column(mia_df, ["MIA", "mia", "MIA Score", "mia_score", "mia_acc"])
    dist_pair = get_first_existing_column(distance_df, ["pair", "Comparison", "comparison"])
    full_col = get_first_existing_column(distance_df, ["full_model_l2", "Distance to Oracle", "distance_to_oracle"])
    if None in [model_col, val_col, mia_model, mia_col, dist_pair, full_col]:
        return

    models = compare_df[model_col].astype(str).tolist()
    acc = compare_df[val_col].tolist()
    mia_map = {str(r[mia_model]): float(r[mia_col]) for _, r in mia_df.iterrows()}
    dist_map = {}
    for _, r in distance_df.iterrows():
        key = str(r[dist_pair])
        if "full" in key.lower():
            dist_map["full"] = float(r[full_col])
        elif "unlearn" in key.lower():
            dist_map["unlearn"] = float(r[full_col])

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    acc_vals = [float(v) for v in acc]
    mia_vals = [mia_map.get(m, np.nan) for m in models]
    dist_vals = []
    for m in models:
        low = m.lower()
        if "full" in low:
            dist_vals.append(dist_map.get("full", np.nan))
        elif "unlearn" in low:
            dist_vals.append(dist_map.get("unlearn", np.nan))
        else:
            dist_vals.append(np.nan)

    ax.bar(x - width, acc_vals, width=width, label="Accuracy", color=COLORS["val"])
    ax.bar(x, mia_vals, width=width, label="MIA", color=COLORS["distance"])
    ax.bar(x + width, dist_vals, width=width, label="Distance", color=COLORS["gap"])
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Combined Performance Summary ({client_name})")
    ax.legend()
    style_axis(ax)
    save_plot(os.path.join(out, f"fig_combined_performance_summary_{client_name}.png"))


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="paper_figures_all")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    setup_paper_style()

    phase2 = load_csv_if_exists("outputs_phase2/metrics/federated_round_metrics.csv")
    ckks = load_csv_if_exists("outputs_ckks/metrics/ckks_round_metrics.csv")

    c0_unlearn = load_csv_if_exists("corrected_results_client0/metrics/unlearn_round_metrics.csv")
    c1_unlearn = load_csv_if_exists("corrected_results_client_1/metrics/unlearn_round_metrics.csv")

    c0_compare = load_csv_if_exists("corrected_results_client0/metrics/phase3_model_comparison.csv")
    c1_compare = load_csv_if_exists("corrected_results_client_1/metrics/phase3_model_comparison.csv")

    c0_mia = load_csv_if_exists("corrected_results_client0/metrics/mia_results.csv")
    c1_mia = load_csv_if_exists("corrected_results_client_1/metrics/mia_results.csv")

    c0_dist = load_csv_if_exists("corrected_results_client0/metrics/distance_summary.csv")
    c1_dist = load_csv_if_exists("corrected_results_client_1/metrics/distance_summary.csv")

    c0_oracle_round = load_csv_if_exists("corrected_results_client0/metrics/oracle_round_metrics.csv")
    c1_oracle_round = load_csv_if_exists("corrected_results_client_1/metrics/oracle_round_metrics.csv")

    c0_unlearn_per_layer = load_csv_if_exists("corrected_results_client0/metrics/unlearn_per_layer_roundwise.csv")
    c1_unlearn_per_layer = load_csv_if_exists("corrected_results_client_1/metrics/unlearn_per_layer_roundwise.csv")

    compare_dfs = {
        "client0": c0_compare,
        "client1": c1_compare,
    }

    # Global
    if phase2 is not None:
        plot_federated_training_accuracy_curve(phase2, args.output_dir)
        plot_federated_training_loss_curve(phase2, args.output_dir)
        plot_global_validation_accuracy_per_round(phase2, args.output_dir)

    if phase2 is not None and ckks is not None:
        plot_global_utility_comparison(phase2, ckks, args.output_dir)

    if ckks is not None:
        plot_training_time_per_round(ckks, args.output_dir)
        plot_encryption_overhead(ckks, args.output_dir)
        plot_communication_cost_per_round(ckks, args.output_dir)
        plot_model_size_overhead_comparison(phase2, ckks, args.output_dir)
        plot_ckks_noise_growth_curve(ckks, args.output_dir)
        plot_accuracy_vs_ckks_noise_level(phase2, ckks, args.output_dir)

    # Multi-client / data distribution
    plot_multi_client_accuracy_comparison(compare_dfs, args.output_dir)
    plot_per_client_accuracy_distribution(compare_dfs, args.output_dir)
    plot_client_data_distribution_heatmap("outputs_ckks/stats/client_indices.json", args.output_dir)
    plot_class_distribution_per_client("outputs_ckks/stats/client_indices.json", args.output_dir)
    plot_accuracy_vs_number_of_clients(compare_dfs, args.output_dir)
    plot_accuracy_vs_dirichlet_alpha(args.output_dir)

    # Client 0
    if c0_compare is not None:
        plot_model_accuracy_comparison(c0_compare, "client0", args.output_dir)
        plot_utility_retention_vs_oracle_gap(c0_compare, "client0", args.output_dir)
    if c0_mia is not None:
        plot_mia_score_comparison(c0_mia, "client0", args.output_dir)
        plot_mia_roc_curve(c0_mia, "client0", args.output_dir)
        plot_mia_auc_comparison(c0_mia, "client0", args.output_dir)
    if c0_unlearn is not None:
        plot_gap(c0_unlearn, "client0", args.output_dir)
        plot_unlearning_validation_accuracy(c0_unlearn, "client0", args.output_dir)
        plot_unlearning_convergence_curve(c0_unlearn, "client0", args.output_dir)
        plot_parameter_update_norm_per_round(c0_unlearn, "client0", args.output_dir)
    if c0_dist is not None:
        plot_fidelity(c0_dist, "client0", args.output_dir)
        plot_full_model_vs_oracle_l2_distance(c0_dist, "client0", args.output_dir)
        plot_last_layer_distance_to_oracle(c0_dist, "client0", args.output_dir)
    if c0_oracle_round is not None and c0_unlearn_per_layer is not None:
        plot_distance_to_oracle_vs_unlearning_rounds(c0_oracle_round, c0_unlearn_per_layer, "client0", args.output_dir)
        plot_layer_wise_distance_to_oracle(c0_unlearn_per_layer, "client0", args.output_dir)
        plot_heatmap(c0_unlearn_per_layer, "client0", args.output_dir)
    if c0_compare is not None and c0_mia is not None:
        plot_accuracy_vs_privacy_tradeoff(c0_compare, c0_mia, "client0", args.output_dir)
    plot_combined_performance_summary(c0_compare, c0_mia, c0_dist, "client0", args.output_dir)

    # Client 1
    if c1_compare is not None:
        plot_model_accuracy_comparison(c1_compare, "client1", args.output_dir)
        plot_utility_retention_vs_oracle_gap(c1_compare, "client1", args.output_dir)
    if c1_mia is not None:
        plot_mia_score_comparison(c1_mia, "client1", args.output_dir)
        plot_mia_roc_curve(c1_mia, "client1", args.output_dir)
        plot_mia_auc_comparison(c1_mia, "client1", args.output_dir)
    if c1_unlearn is not None:
        plot_gap(c1_unlearn, "client1", args.output_dir)
        plot_unlearning_validation_accuracy(c1_unlearn, "client1", args.output_dir)
        plot_unlearning_convergence_curve(c1_unlearn, "client1", args.output_dir)
        plot_parameter_update_norm_per_round(c1_unlearn, "client1", args.output_dir)
    if c1_dist is not None:
        plot_fidelity(c1_dist, "client1", args.output_dir)
        plot_full_model_vs_oracle_l2_distance(c1_dist, "client1", args.output_dir)
        plot_last_layer_distance_to_oracle(c1_dist, "client1", args.output_dir)
    if c1_oracle_round is not None and c1_unlearn_per_layer is not None:
        plot_distance_to_oracle_vs_unlearning_rounds(c1_oracle_round, c1_unlearn_per_layer, "client1", args.output_dir)
        plot_layer_wise_distance_to_oracle(c1_unlearn_per_layer, "client1", args.output_dir)
        plot_heatmap(c1_unlearn_per_layer, "client1", args.output_dir)
    if c1_compare is not None and c1_mia is not None:
        plot_accuracy_vs_privacy_tradeoff(c1_compare, c1_mia, "client1", args.output_dir)
    plot_combined_performance_summary(c1_compare, c1_mia, c1_dist, "client1", args.output_dir)

    print("\nAll requested plots generated where input files were available.")


if __name__ == "__main__":
    main()