import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd

import json
from typing import Dict, List


def load_client_indices(json_path: str) -> Dict[int, List[int]]:
    with open(json_path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

def dirichlet_partition(
    targets: List[int],
    num_clients: int,
    num_classes: int,
    alpha: float,
    min_samples_per_client: int = 10,
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    Partition dataset indices across clients using Dirichlet non-IID allocation.

    Returns:
        client_to_indices: dict[client_id] -> list of sample indices
    """
    rng = np.random.default_rng(seed)
    targets = np.array(targets)

    while True:
        client_indices = [[] for _ in range(num_clients)]

        for cls in range(num_classes):
            cls_indices = np.where(targets == cls)[0]
            rng.shuffle(cls_indices)
            proportions = rng.dirichlet(alpha=np.repeat(0.1, num_clients))
            split_points = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
            split_indices = np.split(cls_indices, split_points)

            for client_id, split in enumerate(split_indices):
                client_indices[client_id].extend(split.tolist())

        sizes = [len(indices) for indices in client_indices]
        if min(sizes) >= min_samples_per_client:
            break

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return {client_id: client_indices[client_id] for client_id in range(num_clients)}


def compute_client_class_distribution(
    targets: List[int],
    client_to_indices: Dict[int, List[int]],
    num_classes: int
) -> Dict[int, Dict[int, int]]:
    """
    Returns:
        dict[client_id][class_id] = count
    """
    distribution = {}
    targets = np.array(targets)

    for client_id, indices in client_to_indices.items():
        client_targets = targets[indices]
        counts = Counter(client_targets.tolist())

        distribution[client_id] = {
            cls: int(counts.get(cls, 0)) for cls in range(num_classes)
        }

    return distribution


def print_client_distribution(distribution: Dict[int, Dict[int, int]], max_classes_to_show: int = 20):
    print("=" * 80)
    print("Client-wise class distribution")
    print("=" * 80)

    for client_id, class_counts in distribution.items():
        total_samples = sum(class_counts.values())
        non_zero = {k: v for k, v in class_counts.items() if v > 0}
        preview_items = list(non_zero.items())[:max_classes_to_show]
        preview_str = ", ".join([f"class_{k}:{v}" for k, v in preview_items])
        print(f"Client {client_id:02d} | total_samples={total_samples} | non_zero_classes={len(non_zero)}")
        print(f"  {preview_str}")
    print("=" * 80)


def save_partition_stats(
    client_to_indices: Dict[int, List[int]],
    distribution: Dict[int, Dict[int, int]],
    save_dir: str
):
    os.makedirs(save_dir, exist_ok=True)

    # Save client indices as JSON
    indices_path = os.path.join(save_dir, "client_indices.json")
    with open(indices_path, "w") as f:
        json.dump({str(k): v for k, v in client_to_indices.items()}, f, indent=2)

    # Save class distribution as JSON
    dist_json_path = os.path.join(save_dir, "client_class_distribution.json")
    with open(dist_json_path, "w") as f:
        json.dump(
            {str(k): {str(c): n for c, n in v.items()} for k, v in distribution.items()},
            f,
            indent=2
        )

    # Save class distribution as CSV
    rows = []
    for client_id, class_counts in distribution.items():
        row = {"client_id": client_id, "total_samples": sum(class_counts.values())}
        for class_id, count in class_counts.items():
            row[f"class_{class_id}"] = count
        rows.append(row)

    df = pd.DataFrame(rows)
    dist_csv_path = os.path.join(save_dir, "client_class_distribution.csv")
    df.to_csv(dist_csv_path, index=False)

    # Save short summary CSV
    summary_rows = []
    for client_id, class_counts in distribution.items():
        non_zero_classes = sum(1 for v in class_counts.values() if v > 0)
        total_samples = sum(class_counts.values())
        summary_rows.append({
            "client_id": client_id,
            "total_samples": total_samples,
            "non_zero_classes": non_zero_classes
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(save_dir, "client_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"Saved partition stats to: {save_dir}")