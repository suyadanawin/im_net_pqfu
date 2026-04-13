import os
import csv
from typing import Dict, List

import matplotlib.pyplot as plt


def load_combined_results(csv_path: str, client_id: int) -> List[Dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "client_id": client_id,
                "model": row["model"],
                "val_acc": float(row["val_acc"]),
                "forget_acc": float(row["forget_acc"]),
                "retain_acc": float(row["retain_acc"]),
                "mia_acc": float(row["mia_acc"]),
                "distance_to_oracle": float(row["distance_to_oracle"]),
            })
    return rows


def collect_all_results(base_dirs: Dict[int, str]) -> List[Dict]:
    all_rows = []
    for client_id, base_dir in base_dirs.items():
        csv_path = os.path.join(base_dir, "metrics", "final_combined_results.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Missing combined results file for client {client_id}: {csv_path}"
            )
        all_rows.extend(load_combined_results(csv_path, client_id))
    return all_rows


def group_metric_by_client_and_model(
    rows: List[Dict],
    metric_name: str,
    client_ids: List[int],
    model_order: List[str],
) -> Dict[str, List[float]]:
    grouped = {model: [] for model in model_order}

    for client_id in client_ids:
        client_rows = [r for r in rows if r["client_id"] == client_id]
        for model in model_order:
            match = next((r for r in client_rows if r["model"] == model), None)
            if match is None:
                raise ValueError(f"Missing model '{model}' for client {client_id}")
            grouped[model].append(match[metric_name])

    return grouped


def plot_grouped_bar(
    grouped_data: Dict[str, List[float]],
    client_ids: List[int],
    title: str,
    ylabel: str,
    output_path: str,
) -> None:
    model_order = list(grouped_data.keys())
    n_clients = len(client_ids)
    x = list(range(n_clients))
    width = 0.22

    offsets = {
        model_order[0]: -width,
        model_order[1]: 0.0,
        model_order[2]: width,
    }

    plt.figure(figsize=(10, 6))

    for model in model_order:
        positions = [xi + offsets[model] for xi in x]
        values = grouped_data[model]
        plt.bar(positions, values, width=width, label=model.capitalize())

        for px, val in zip(positions, values):
            plt.text(px, val + 0.3, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, [f"Client {cid}" for cid in client_ids])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_summary_csv(rows: List[Dict], output_csv: str) -> None:
    fieldnames = [
        "client_id",
        "model",
        "val_acc",
        "forget_acc",
        "retain_acc",
        "mia_acc",
        "distance_to_oracle",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # Adjust these folder names if needed
    base_dirs = {
        0: "corrected_results_client_0",
        1: "corrected_results_client_1",
        5: "corrected_results_client_5",
    }

    output_dir = "paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    rows = collect_all_results(base_dirs)
    save_summary_csv(rows, os.path.join(output_dir, "phase3_all_clients_summary.csv"))

    client_ids = [0, 1, 5]
    model_order = ["full", "oracle", "unlearn"]

    # Figure 1: Validation Accuracy
    val_grouped = group_metric_by_client_and_model(
        rows, "val_acc", client_ids, model_order
    )
    plot_grouped_bar(
        grouped_data=val_grouped,
        client_ids=client_ids,
        title="Global Validation Accuracy Across Forget Clients",
        ylabel="Validation Accuracy (%)",
        output_path=os.path.join(output_dir, "figure1_validation_accuracy.png"),
    )

    # Figure 2: Forget Accuracy
    forget_grouped = group_metric_by_client_and_model(
        rows, "forget_acc", client_ids, model_order
    )
    plot_grouped_bar(
        grouped_data=forget_grouped,
        client_ids=client_ids,
        title="Forget-Subset Accuracy Across Forget Clients",
        ylabel="Forget Accuracy (%)",
        output_path=os.path.join(output_dir, "figure2_forget_accuracy.png"),
    )

    # Figure 3: MIA Accuracy
    mia_grouped = group_metric_by_client_and_model(
        rows, "mia_acc", client_ids, model_order
    )
    plot_grouped_bar(
        grouped_data=mia_grouped,
        client_ids=client_ids,
        title="Membership Inference Attack Accuracy Across Forget Clients",
        ylabel="MIA Accuracy",
        output_path=os.path.join(output_dir, "figure3_mia_accuracy.png"),
    )

    # Figure 4: Distance to Oracle
    dist_grouped = group_metric_by_client_and_model(
        rows, "distance_to_oracle", client_ids, model_order
    )
    plot_grouped_bar(
        grouped_data=dist_grouped,
        client_ids=client_ids,
        title="Distance to Oracle Across Forget Clients",
        ylabel="L2 Distance to Oracle",
        output_path=os.path.join(output_dir, "figure4_distance_to_oracle.png"),
    )

    print("Saved summary CSV and all figures to:", output_dir)
    print("Files created:")
    print("- paper_figures/phase3_all_clients_summary.csv")
    print("- paper_figures/figure1_validation_accuracy.png")
    print("- paper_figures/figure2_forget_accuracy.png")
    print("- paper_figures/figure3_mia_accuracy.png")
    print("- paper_figures/figure4_distance_to_oracle.png")


if __name__ == "__main__":
    main()