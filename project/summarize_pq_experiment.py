import json
import os
import pandas as pd

SETTINGS = {
    "PQ-Low (4096)": "./outputs_pq_4096",
    "PQ-Mid (8192)": "./outputs_pq_8192",
    "PQ-High (16384)": "./outputs_pq_16384",
}

rows = []

for setting_name, root in SETTINGS.items():
    ckks_metrics_csv = os.path.join(root, "ckks", "metrics", "ckks_round_metrics.csv")
    ckks_run_json = os.path.join(root, "ckks", "stats", "run_summary.json")
    dist_json = os.path.join(root, "eval", "ckks_distance_summary.json")
    eval_csv = os.path.join(root, "eval", "ckks_unlearning_model_comparison.csv")
    mia_csv = os.path.join(root, "mia", "mia_ckks_unlearning_summary.csv")

    row = {
        "setting": setting_name,
        "best_val_acc": None,
        "final_val_acc": None,
        "mean_encrypt_time_sec": None,
        "mean_agg_time_sec": None,
        "mean_decrypt_time_sec": None,
        "mean_ciphertext_bytes": None,
        "unlearn_vs_oracle_l2": None,
        "unlearn_vs_oracle_last_layer_l2": None,
        "mia_acc": None,
    }

    if os.path.exists(ckks_metrics_csv):
        df = pd.read_csv(ckks_metrics_csv)
        if "global_val_acc" in df.columns:
            row["best_val_acc"] = float(df["global_val_acc"].max())
            row["final_val_acc"] = float(df["global_val_acc"].iloc[-1])
        if "total_encrypt_time_sec" in df.columns:
            row["mean_encrypt_time_sec"] = float(df["total_encrypt_time_sec"].mean())
        if "aggregation_time_sec" in df.columns:
            row["mean_agg_time_sec"] = float(df["aggregation_time_sec"].mean())
        if "decrypt_time_sec" in df.columns:
            row["mean_decrypt_time_sec"] = float(df["decrypt_time_sec"].mean())
        if "total_ciphertext_bytes" in df.columns:
            row["mean_ciphertext_bytes"] = float(df["total_ciphertext_bytes"].mean())

    if os.path.exists(dist_json):
        with open(dist_json, "r", encoding="utf-8") as f:
            d = json.load(f)
        row["unlearn_vs_oracle_l2"] = d.get("unlearn_vs_oracle_l2", None)
        row["unlearn_vs_oracle_last_layer_l2"] = d.get("unlearn_vs_oracle_last_layer_l2", None)

    if os.path.exists(mia_csv):
        df_mia = pd.read_csv(mia_csv)
        target = df_mia[df_mia["model"] == "unlearn_ckks"]
        if len(target) > 0:
            row["mia_acc"] = float(target["mia_acc"].iloc[0])

    rows.append(row)

out_df = pd.DataFrame(rows)
os.makedirs("./outputs_pq_summary", exist_ok=True)
out_csv = "./outputs_pq_summary/pq_security_tradeoff_summary.csv"
out_df.to_csv(out_csv, index=False)
print(out_df)
print(f"[SAVED] {out_csv}")