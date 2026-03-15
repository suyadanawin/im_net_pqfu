import pandas as pd
import matplotlib.pyplot as plt
import json
import glob
import os

# 1. Find the latest metrics file
log_dir = "outputs/logs"
files = glob.glob(os.path.join(log_dir, "*.json"))
if not files:
    print(f"No JSON files found in {log_dir}!")
    exit()

latest_file = max(files, key=os.path.getctime)
print(f"Plotting results from: {latest_file}")

# 2. Load the data
with open(latest_file, 'r') as f:
    data = json.load(f)

# 3. Process data into a table
df = pd.DataFrame(data)
df['epoch'] = range(1, len(df) + 1) # Create the epoch column

# 4. Create the Plot
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', marker='o')
plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='red', marker='s')
plt.title('Phase 1: Loss Trends')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['train_acc'], label='Train Acc', color='green', marker='o')
plt.plot(df['epoch'], df['val_acc'], label='Val Acc', color='orange', marker='s')
plt.title('Phase 1: Accuracy Trends')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

# 5. Save and Show
save_path = 'outputs/phase1_visualization.png'
plt.savefig(save_path)
print(f"Success! Graph saved to {save_path}")
plt.show()