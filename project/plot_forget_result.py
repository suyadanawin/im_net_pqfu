import matplotlib.pyplot as plt
import numpy as np

# Data from results
forget_ids = ['ID 0', 'ID 1', 'ID 5']
full_acc = [44.5, 44.5, 44.5]    # Approximated from 'full' total
oracle_acc = [18.9, 18.9, 18.9]  # Approximated from 'oracle' total
unlearn_acc = [23.1, 23.1, 23.1] # 'unlearn' result

x = np.arange(len(forget_ids))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, full_acc, width, label='Full Model (Before)', color='#ff7f0e')
ax.bar(x, oracle_acc, width, label='Oracle (Ideal)', color='#2ca02c')
ax.bar(x + width, unlearn_acc, width, label='Unlearn (Current)', color='#1f77b4')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Forget Set Performance: IDs 0, 1, 5')
ax.set_xticks(x)
ax.set_xticklabels(forget_ids)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.1)

plt.savefig('forget_comparison.png')
print("Visual saved as forget_comparison.png")