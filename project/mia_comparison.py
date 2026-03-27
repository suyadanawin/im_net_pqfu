import matplotlib.pyplot as plt

models = ['Full', 'Oracle', 'Unlearn']
mia_accuracies = [0.488, 0.450, 0.434]

plt.figure(figsize=(8, 5))
colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
bars = plt.bar(models, mia_accuracies, color=colors)

plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess (0.5)')
plt.ylabel('MIA Accuracy')
plt.title('Privacy Check: Membership Inference Attack')
plt.ylim(0, 0.6)
plt.legend()

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, yval, ha='center')

plt.savefig('mia_comparison.png')
print("MIA Plot saved!")