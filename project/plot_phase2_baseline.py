import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# Load your CSV
# ==============================
# Get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load CSV from same folder as script
csv_path = os.path.join(BASE_DIR, "fl_baseline_metrics.csv")

df = pd.read_csv(csv_path)
# Create output folder
os.makedirs("phase2_figures", exist_ok=True)

# ==============================
# 1. Global Validation Accuracy (MAIN FIGURE)
# ==============================
plt.figure(figsize=(8,5))
plt.plot(df["round"], df["global_val_acc"], marker='o')
plt.xlabel("Federated Rounds")
plt.ylabel("Validation Accuracy (%)")
plt.title("Federated Learning Convergence (Tiny-ImageNet, Non-IID)")
plt.grid(True)
plt.savefig("phase2_figures/global_val_accuracy.png", dpi=300)
plt.close()

# ==============================
# 2. Global Validation Loss
# ==============================
plt.figure(figsize=(8,5))
plt.plot(df["round"], df["global_val_loss"], marker='o')
plt.xlabel("Federated Rounds")
plt.ylabel("Validation Loss")
plt.title("Validation Loss over Federated Rounds")
plt.grid(True)
plt.savefig("phase2_figures/global_val_loss.png", dpi=300)
plt.close()

# ==============================
# 3. Train vs Validation Accuracy
# ==============================
plt.figure(figsize=(8,5))
plt.plot(df["round"], df["avg_client_train_acc"], label="Train Accuracy")
plt.plot(df["round"], df["global_val_acc"], label="Validation Accuracy")
plt.xlabel("Federated Rounds")
plt.ylabel("Accuracy (%)")
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("phase2_figures/train_vs_val_accuracy.png", dpi=300)
plt.close()

# ==============================
# 4. Client Train Accuracy
# ==============================
plt.figure(figsize=(8,5))
plt.plot(df["round"], df["avg_client_train_acc"], marker='o')
plt.xlabel("Federated Rounds")
plt.ylabel("Client Train Accuracy (%)")
plt.title("Average Client Training Accuracy")
plt.grid(True)
plt.savefig("phase2_figures/client_train_accuracy.png", dpi=300)
plt.close()

print("✅ Phase 2 plots saved in 'phase2_figures'")