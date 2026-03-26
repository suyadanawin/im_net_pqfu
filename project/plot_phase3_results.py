import os
import pandas as pd
import matplotlib.pyplot as plt
from src.phase3_utils import load_config, ensure_dirs


def main():
    config = load_config("project/config3.yaml")
    ensure_dirs([config["paths"]["plots_dir"]])

    metrics_dir = config["paths"]["metrics_dir"]
    plots_dir = config["paths"]["plots_dir"]

    comp = pd.read_csv(os.path.join(metrics_dir, "phase3_model_comparison.csv"))
    plt.figure(figsize=(6, 4))
    plt.bar(comp["model"], comp["val_acc"])
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Full vs Oracle vs Unlearned")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase3_accuracy_compare.png"))
    plt.close()

    dist = pd.read_csv(os.path.join(metrics_dir, "distance_summary.csv"))
    plt.figure(figsize=(6, 4))
    plt.bar(dist["pair"], dist["full_model_l2"])
    plt.ylabel("L2 Distance")
    plt.title("Distance to Oracle")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase3_distance_to_oracle.png"))
    plt.close()

    fr = pd.read_csv(os.path.join(metrics_dir, "forget_retain_results.csv"))
    pivot = fr.pivot(index="model", columns="split", values="acc")
    pivot.plot(kind="bar", figsize=(7, 4))
    plt.ylabel("Accuracy (%)")
    plt.title("Forget vs Retain Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase3_forget_retain.png"))
    plt.close()

    mia = pd.read_csv(os.path.join(metrics_dir, "mia_results.csv"))
    plt.figure(figsize=(6, 4))
    plt.bar(mia["model"], mia["mia_acc"])
    plt.ylabel("MIA Accuracy")
    plt.title("Membership Inference Attack")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase3_mia.png"))
    plt.close()

    print("Saved all plots.")


if __name__ == "__main__":
    main()