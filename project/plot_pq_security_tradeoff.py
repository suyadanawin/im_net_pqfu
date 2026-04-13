import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "./outputs_pq_summary/pq_security_tradeoff_summary.csv"
df = pd.read_csv(csv_path)

os.makedirs("./outputs_pq_summary/plots", exist_ok=True)

# Accuracy plot
plt.figure(figsize=(7, 5))
plt.plot(df["setting"], df["best_val_acc"], marker="o")
plt.ylabel("Best Validation Accuracy (%)")
plt.xlabel("CKKS Security Setting")
plt.title("Security Level vs Validation Accuracy")
plt.tight_layout()
plt.savefig("./outputs_pq_summary/plots/security_vs_accuracy.png", dpi=300)
plt.close()

# Encryption time plot
plt.figure(figsize=(7, 5))
plt.plot(df["setting"], df["mean_encrypt_time_sec"], marker="o")
plt.ylabel("Mean Encryption Time per Round (s)")
plt.xlabel("CKKS Security Setting")
plt.title("Security Level vs Encryption Time")
plt.tight_layout()
plt.savefig("./outputs_pq_summary/plots/security_vs_encrypt_time.png", dpi=300)
plt.close()

# Ciphertext size plot
plt.figure(figsize=(7, 5))
plt.plot(df["setting"], df["mean_ciphertext_bytes"], marker="o")
plt.ylabel("Mean Ciphertext Bytes per Round")
plt.xlabel("CKKS Security Setting")
plt.title("Security Level vs Ciphertext Size")
plt.tight_layout()
plt.savefig("./outputs_pq_summary/plots/security_vs_ciphertext_size.png", dpi=300)
plt.close()

# Oracle distance plot
plt.figure(figsize=(7, 5))
plt.plot(df["setting"], df["unlearn_vs_oracle_l2"], marker="o")
plt.ylabel("Unlearn vs Oracle L2 Distance")
plt.xlabel("CKKS Security Setting")
plt.title("Security Level vs Oracle Distance")
plt.tight_layout()
plt.savefig("./outputs_pq_summary/plots/security_vs_oracle_distance.png", dpi=300)
plt.close()

print("[SAVED] plots in ./outputs_pq_summary/plots/")