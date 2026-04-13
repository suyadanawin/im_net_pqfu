import csv
import os

metrics_dir = "corrected_results_client_1/metrics"

# Load validation results
val_data = {}
with open(os.path.join(metrics_dir, "phase3_model_comparison.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        val_data[row["model"]] = row

# Load forget/retain
fr_data = {}
with open(os.path.join(metrics_dir, "forget_retain_train_subset_results.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["model"], row["split"])
        fr_data[key] = row

# Load MIA
mia_data = {}
with open(os.path.join(metrics_dir, "mia_results.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        mia_data[row["model"]] = row

# Load distance
dist_data = {}
with open(os.path.join(metrics_dir, "distance_summary.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        dist_data[row["pair"]] = row

# Combine
rows = []
for model in ["full", "oracle", "unlearn"]:
    row = {
        "model": model,

        "val_acc": val_data[model]["val_acc"],

        "forget_acc": fr_data[(model, "forget_train_subset")]["acc"],
        "retain_acc": fr_data[(model, "retain_train_subset")]["acc"],

        "mia_acc": mia_data[model]["mia_acc"],

        "distance_to_oracle":
            dist_data["unlearn_vs_oracle"]["full_model_l2"]
            if model == "unlearn"
            else dist_data["full_vs_oracle"]["full_model_l2"]
    }
    rows.append(row)

# Save
out_file = os.path.join(metrics_dir, "client1_final_results.csv")
with open(out_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Saved:", out_file)