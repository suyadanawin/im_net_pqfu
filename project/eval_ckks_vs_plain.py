import sys
import os
sys.path.append(os.path.abspath("."))

import argparse
import json

import pandas as pd

from src.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain_csv", type=str, required=True)
    parser.add_argument("--ckks_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    if not os.path.exists(args.plain_csv):
        raise FileNotFoundError(f"Plain metrics CSV not found: {args.plain_csv}")

    if not os.path.exists(args.ckks_csv):
        raise FileNotFoundError(f"CKKS metrics CSV not found: {args.ckks_csv}")

    plain_df = pd.read_csv(args.plain_csv)
    ckks_df = pd.read_csv(args.ckks_csv)

    required_plain_cols = ["global_val_acc"]
    required_ckks_cols = [
        "global_val_acc",
        "total_encrypt_time_sec",
        "aggregation_time_sec",
        "decrypt_time_sec",
    ]

    for col in required_plain_cols:
        if col not in plain_df.columns:
            raise ValueError(f"Missing column in plain CSV: {col}")

    for col in required_ckks_cols:
        if col not in ckks_df.columns:
            raise ValueError(f"Missing column in CKKS CSV: {col}")

    summary = {
        "plain_best_val_acc": float(plain_df["global_val_acc"].max()),
        "ckks_best_val_acc": float(ckks_df["global_val_acc"].max()),
        "plain_last_val_acc": float(plain_df["global_val_acc"].iloc[-1]),
        "ckks_last_val_acc": float(ckks_df["global_val_acc"].iloc[-1]),
        "accuracy_gap_best": float(
            plain_df["global_val_acc"].max() - ckks_df["global_val_acc"].max()
        ),
        "accuracy_gap_last": float(
            plain_df["global_val_acc"].iloc[-1] - ckks_df["global_val_acc"].iloc[-1]
        ),
        "ckks_mean_encrypt_time_sec": float(
            ckks_df["total_encrypt_time_sec"].mean()
        ),
        "ckks_mean_aggregation_time_sec": float(
            ckks_df["aggregation_time_sec"].mean()
        ),
        "ckks_mean_decrypt_time_sec": float(
            ckks_df["decrypt_time_sec"].mean()
        ),
    }

    if "total_ciphertext_bytes" in ckks_df.columns:
        summary["ckks_mean_ciphertext_bytes"] = float(
            ckks_df["total_ciphertext_bytes"].mean()
        )
        summary["ckks_max_ciphertext_bytes"] = float(
            ckks_df["total_ciphertext_bytes"].max()
        )

    # Save JSON
    summary_path = os.path.join(args.output_dir, "ckks_vs_plain_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ✅ Save CSV (NEW)
    csv_path = os.path.join(args.output_dir, "ckks_vs_plain_summary.csv")
    df = pd.DataFrame([summary])
    df.to_csv(csv_path, index=False)

    print("Saved JSON:", summary_path)
    print("Saved CSV:", csv_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()