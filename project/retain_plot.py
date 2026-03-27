import matplotlib.pyplot as plt
import numpy as np

# Data from your latest terminal output
labels = ['Retain Set Accuracy']
full_acc = [44.60]
oracle_acc = [45.87]
unlearn_acc = [61.60]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(8, 6))

# Creating the bars
rects1 = ax.bar(x - width, full_acc, width, label='Full Model', color='#ff7f0e')
rects2 = ax.bar(x, oracle_acc, width, label='Oracle Model', color='#2ca02c')
rects3 = ax.bar(x + width, unlearn_acc, width, label='Unlearn Model', color='#1f77b4')

# Add text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_title('Utility Preservation: Retain Set Performance')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.savefig('retain_comparison.png')
print("Visual saved as retain_comparison.png")