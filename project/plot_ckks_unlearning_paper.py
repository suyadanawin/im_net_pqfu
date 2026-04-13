import os
import json
import argparse
import textwrap

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_bar_plot(labels, values, ylabel, title, out_path, annotate_fmt="{:.2f}", rotate=0):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    if rotate != 0:
        plt.xticks(rotation=rotate, ha="right")

    for bar, v in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            annotate_fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def save_line_plot(x, y, xlabel, ylabel, title, out_path, annotate_last=True):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if annotate_last and len(x) > 0:
        plt.text(x[-1], y[-1], f"{y[-1]:.2f}", fontsize=10, va="bottom", ha="left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def save_dual_line_plot(x, y1, y2, label1, label2, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y1, marker="o", label=label1)
    plt.plot(x, y2, marker="s", label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="./outputs_ckks_unlearning",
        help="Root folder for CKKS unlearning outputs",
    )
    parser.add_argument(
        "--fl_root",
        type=str,
        default="./outputs_ckks",
        help="Root folder for baseline CKKS FL outputs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_ckks_unlearning/plots_ckks_unlearning",
        help="Output folder for generated plots",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # -----------------------------
    # Input file paths
    # -----------------------------
    eval_csv = os.path.join(args.root, "eval", "ckks_unlearning_model_comparison.csv")
    dist_json = os.path.join(args.root, "eval", "ckks_distance_summary.json")
    mia_csv = os.path.join(args.root, "mia", "mia_ckks_unlearning_summary.csv")

    ckks_fl_round_csv = os.path.join(args.fl_root, "metrics", "ckks_round_metrics.csv")
    ckks_unlearn_round_csv = os.path.join(args.root, "unlearn", "metrics", "ckks_unlearn_round_metrics.csv")

    # -----------------------------
    # Load files
    # -----------------------------
    eval_df = load_csv(eval_csv)
    dist_data = load_json(dist_json)
    mia_df = load_csv(mia_csv)

    ckks_fl_df = None
    if os.path.exists(ckks_fl_round_csv):
        ckks_fl_df = pd.read_csv(ckks_fl_round_csv)

    ckks_unlearn_df = None
    if os.path.exists(ckks_unlearn_round_csv):
        ckks_unlearn_df = pd.read_csv(ckks_unlearn_round_csv)

    # -----------------------------
    # Normalize model order
    # -----------------------------
    desired_order = ["full_ckks", "oracle_ckks", "unlearn_ckks"]
    eval_df["model"] = pd.Categorical(eval_df["model"], categories=desired_order, ordered=True)
    eval_df = eval_df.sort_values("model").reset_index(drop=True)

    mia_df["model"] = pd.Categorical(mia_df["model"], categories=desired_order, ordered=True)
    mia_df = mia_df.sort_values("model").reset_index(drop=True)

    model_labels = {
        "full_ckks": "Full CKKS",
        "oracle_ckks": "Oracle CKKS",
        "unlearn_ckks": "Unlearn CKKS",
    }

    labels = [model_labels[m] for m in eval_df["model"].tolist()]

    # -----------------------------
    # Plot 1: Validation Accuracy Comparison
    # -----------------------------
    save_bar_plot(
        labels=labels,
        values=eval_df["val_acc"].tolist(),
        ylabel="Validation Accuracy (%)",
        title="CKKS Model Comparison: Validation Accuracy",
        out_path=os.path.join(args.output_dir, "fig_ckks_val_accuracy_comparison.png"),
    )

    # -----------------------------
    # Plot 2: Forget Accuracy Comparison
    # -----------------------------
    save_bar_plot(
        labels=labels,
        values=eval_df["forget_acc"].tolist(),
        ylabel="Forgotten Client Accuracy (%)",
        title="CKKS Model Comparison: Forgotten Client Accuracy",
        out_path=os.path.join(args.output_dir, "fig_ckks_forget_accuracy_comparison.png"),
    )

    # -----------------------------
    # Plot 3: Retain Accuracy Comparison
    # -----------------------------
    save_bar_plot(
        labels=labels,
        values=eval_df["retain_acc"].tolist(),
        ylabel="Retained Data Accuracy (%)",
        title="CKKS Model Comparison: Retained Data Accuracy",
        out_path=os.path.join(args.output_dir, "fig_ckks_retain_accuracy_comparison.png"),
    )

    # -----------------------------
    # Plot 4: Combined Accuracy Comparison
    # -----------------------------
    x = labels
    plt.figure(figsize=(10, 6))
    width = 0.25
    positions = list(range(len(x)))

    val_vals = eval_df["val_acc"].tolist()
    forget_vals = eval_df["forget_acc"].tolist()
    retain_vals = eval_df["retain_acc"].tolist()

    plt.bar([p - width for p in positions], val_vals, width=width, label="Validation Acc")
    plt.bar(positions, forget_vals, width=width, label="Forget Acc")
    plt.bar([p + width for p in positions], retain_vals, width=width, label="Retain Acc")

    plt.xticks(positions, x)
    plt.ylabel("Accuracy (%)")
    plt.title("CKKS Model Comparison: Validation / Forget / Retain Accuracy")
    plt.legend()
    plt.tight_layout()
    combined_acc_path = os.path.join(args.output_dir, "fig_ckks_combined_accuracy_comparison.png")
    plt.savefig(combined_acc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {combined_acc_path}")

    # -----------------------------
    # Plot 5: Distance to Oracle
    # -----------------------------
    dist_labels = ["Full vs Oracle", "Unlearn vs Oracle"]
    dist_vals = [
        dist_data["full_vs_oracle_l2"],
        dist_data["unlearn_vs_oracle_l2"],
    ]
    save_bar_plot(
        labels=dist_labels,
        values=dist_vals,
        ylabel="L2 Distance",
        title="CKKS Distance to Oracle Model",
        out_path=os.path.join(args.output_dir, "fig_ckks_distance_to_oracle.png"),
    )

    # -----------------------------
    # Plot 6: Last-layer Distance to Oracle
    # -----------------------------
    last_layer_vals = [
        dist_data["full_vs_oracle_last_layer_l2"],
        dist_data["unlearn_vs_oracle_last_layer_l2"],
    ]
    save_bar_plot(
        labels=dist_labels,
        values=last_layer_vals,
        ylabel="Last-layer L2 Distance",
        title="CKKS Last-layer Distance to Oracle",
        out_path=os.path.join(args.output_dir, "fig_ckks_last_layer_distance_to_oracle.png"),
    )

    # -----------------------------
    # Plot 7: MIA Accuracy Comparison
    # -----------------------------
    mia_labels = [model_labels[m] for m in mia_df["model"].tolist()]
    save_bar_plot(
        labels=mia_labels,
        values=mia_df["mia_acc"].tolist(),
        ylabel="MIA Accuracy",
        title="CKKS Membership Inference Attack Comparison",
        out_path=os.path.join(args.output_dir, "fig_ckks_mia_accuracy_comparison.png"),
        annotate_fmt="{:.4f}",
    )

    # -----------------------------
    # Plot 8: Member vs Nonmember Confidence
    # -----------------------------
    plt.figure(figsize=(10, 6))
    width = 0.35
    positions = list(range(len(mia_labels)))
    member_conf = mia_df["member_mean_conf"].tolist()
    nonmember_conf = mia_df["nonmember_mean_conf"].tolist()

    plt.bar([p - width / 2 for p in positions], member_conf, width=width, label="Member Mean Confidence")
    plt.bar([p + width / 2 for p in positions], nonmember_conf, width=width, label="Nonmember Mean Confidence")

    plt.xticks(positions, mia_labels)
    plt.ylabel("Confidence")
    plt.title("CKKS MIA: Member vs Nonmember Mean Confidence")
    plt.legend()
    plt.tight_layout()
    conf_path = os.path.join(args.output_dir, "fig_ckks_member_nonmember_confidence.png")
    plt.savefig(conf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {conf_path}")

    # -----------------------------
    # Plot 9: CKKS FL Accuracy Curve
    # -----------------------------
    if ckks_fl_df is not None and "round" in ckks_fl_df.columns and "global_val_acc" in ckks_fl_df.columns:
        save_line_plot(
            x=ckks_fl_df["round"].tolist(),
            y=ckks_fl_df["global_val_acc"].tolist(),
            xlabel="Communication Round",
            ylabel="Validation Accuracy (%)",
            title="CKKS FL Validation Accuracy per Round",
            out_path=os.path.join(args.output_dir, "fig_ckks_fl_val_accuracy_curve.png"),
        )

    # -----------------------------
    # Plot 10: CKKS Unlearning Accuracy Curve
    # -----------------------------
    if ckks_unlearn_df is not None and "round" in ckks_unlearn_df.columns and "global_val_acc" in ckks_unlearn_df.columns:
        save_line_plot(
            x=ckks_unlearn_df["round"].tolist(),
            y=ckks_unlearn_df["global_val_acc"].tolist(),
            xlabel="Unlearning Round",
            ylabel="Validation Accuracy (%)",
            title="CKKS Unlearning Validation Accuracy per Round",
            out_path=os.path.join(args.output_dir, "fig_ckks_unlearning_val_accuracy_curve.png"),
        )

    # -----------------------------
    # Plot 11: CKKS FL vs Unlearning Accuracy Curve
    # -----------------------------
    if (
        ckks_fl_df is not None
        and ckks_unlearn_df is not None
        and "round" in ckks_fl_df.columns
        and "round" in ckks_unlearn_df.columns
        and "global_val_acc" in ckks_fl_df.columns
        and "global_val_acc" in ckks_unlearn_df.columns
    ):
        save_dual_line_plot(
            x=ckks_unlearn_df["round"].tolist(),
            y1=ckks_fl_df["global_val_acc"].tolist()[: len(ckks_unlearn_df)],
            y2=ckks_unlearn_df["global_val_acc"].tolist(),
            label1="CKKS FL",
            label2="CKKS Unlearning",
            xlabel="Round",
            ylabel="Validation Accuracy (%)",
            title="CKKS FL vs CKKS Unlearning Validation Accuracy",
            out_path=os.path.join(args.output_dir, "fig_ckks_fl_vs_unlearning_accuracy_curve.png"),
        )

    # -----------------------------
    # Plot 12: CKKS FL Runtime Breakdown
    # -----------------------------
    if ckks_fl_df is not None:
        required = {"round", "total_encrypt_time_sec", "aggregation_time_sec", "decrypt_time_sec"}
        if required.issubset(set(ckks_fl_df.columns)):
            plt.figure(figsize=(10, 6))
            plt.plot(ckks_fl_df["round"], ckks_fl_df["total_encrypt_time_sec"], marker="o", label="Encrypt")
            plt.plot(ckks_fl_df["round"], ckks_fl_df["aggregation_time_sec"], marker="s", label="Aggregate")
            plt.plot(ckks_fl_df["round"], ckks_fl_df["decrypt_time_sec"], marker="^", label="Decrypt")
            plt.xlabel("Communication Round")
            plt.ylabel("Time (sec)")
            plt.title("CKKS FL Runtime Breakdown per Round")
            plt.legend()
            plt.tight_layout()
            out_path = os.path.join(args.output_dir, "fig_ckks_fl_runtime_breakdown.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[SAVED] {out_path}")

    # -----------------------------
    # Plot 13: CKKS Unlearning Runtime Breakdown
    # -----------------------------
    if ckks_unlearn_df is not None:
        required = {"round", "total_encrypt_time_sec", "aggregation_time_sec", "decrypt_time_sec"}
        if required.issubset(set(ckks_unlearn_df.columns)):
            plt.figure(figsize=(10, 6))
            plt.plot(ckks_unlearn_df["round"], ckks_unlearn_df["total_encrypt_time_sec"], marker="o", label="Encrypt")
            plt.plot(ckks_unlearn_df["round"], ckks_unlearn_df["aggregation_time_sec"], marker="s", label="Aggregate")
            plt.plot(ckks_unlearn_df["round"], ckks_unlearn_df["decrypt_time_sec"], marker="^", label="Decrypt")
            plt.xlabel("Unlearning Round")
            plt.ylabel("Time (sec)")
            plt.title("CKKS Unlearning Runtime Breakdown per Round")
            plt.legend()
            plt.tight_layout()
            out_path = os.path.join(args.output_dir, "fig_ckks_unlearning_runtime_breakdown.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[SAVED] {out_path}")

    # -----------------------------
    # Plot 14: Ciphertext Size per Round (FL)
    # -----------------------------
    if ckks_fl_df is not None and {"round", "total_ciphertext_bytes"}.issubset(set(ckks_fl_df.columns)):
        save_line_plot(
            x=ckks_fl_df["round"].tolist(),
            y=ckks_fl_df["total_ciphertext_bytes"].tolist(),
            xlabel="Communication Round",
            ylabel="Ciphertext Bytes",
            title="CKKS FL Ciphertext Size per Round",
            out_path=os.path.join(args.output_dir, "fig_ckks_fl_ciphertext_size_curve.png"),
            annotate_last=False,
        )

    # -----------------------------
    # Plot 15: Ciphertext Size per Round (Unlearning)
    # -----------------------------
    if ckks_unlearn_df is not None and {"round", "total_ciphertext_bytes"}.issubset(set(ckks_unlearn_df.columns)):
        save_line_plot(
            x=ckks_unlearn_df["round"].tolist(),
            y=ckks_unlearn_df["total_ciphertext_bytes"].tolist(),
            xlabel="Unlearning Round",
            ylabel="Ciphertext Bytes",
            title="CKKS Unlearning Ciphertext Size per Round",
            out_path=os.path.join(args.output_dir, "fig_ckks_unlearning_ciphertext_size_curve.png"),
            annotate_last=False,
        )

    # -----------------------------
    # Save one summary table too
    # -----------------------------
    summary_rows = []

    for _, row in eval_df.iterrows():
        summary_rows.append({
            "model": row["model"],
            "val_acc": row["val_acc"],
            "forget_acc": row["forget_acc"],
            "retain_acc": row["retain_acc"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_dir, "table_ckks_unlearning_accuracy_summary.csv"), index=False)
    print(f"[SAVED] {os.path.join(args.output_dir, 'table_ckks_unlearning_accuracy_summary.csv')}")

    dist_df = pd.DataFrame([
        {
            "full_vs_oracle_l2": dist_data["full_vs_oracle_l2"],
            "unlearn_vs_oracle_l2": dist_data["unlearn_vs_oracle_l2"],
            "full_vs_oracle_last_layer_l2": dist_data["full_vs_oracle_last_layer_l2"],
            "unlearn_vs_oracle_last_layer_l2": dist_data["unlearn_vs_oracle_last_layer_l2"],
        }
    ])
    dist_df.to_csv(os.path.join(args.output_dir, "table_ckks_unlearning_distance_summary.csv"), index=False)
    print(f"[SAVED] {os.path.join(args.output_dir, 'table_ckks_unlearning_distance_summary.csv')}")

    mia_df.to_csv(os.path.join(args.output_dir, "table_ckks_unlearning_mia_summary.csv"), index=False)
    print(f"[SAVED] {os.path.join(args.output_dir, 'table_ckks_unlearning_mia_summary.csv')}")

    print("\nAll CKKS unlearning plots saved successfully.")


if __name__ == "__main__":
    main()