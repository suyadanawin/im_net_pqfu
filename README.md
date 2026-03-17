# IM_NET_PQFU
## Phase 1: Environment & Containerization, Dataset Preparation & Preprocessing
"In Phase 1, I focused on ensuring the Tiny-ImageNet validation set was correctly restructured to provide accurate metrics, and I validated the $\alpha=0.1$ Dirichlet partitioning to ensure we have a sufficiently challenging Non-IID environment for the PQC evaluation."
## Phase 2: Federated Training (Large-Scale Baseline)
In this phase, we established a robust global baseline model using a 200-class dataset to serve as the foundation for Certified Unlearning evaluations.

### Experiment Setup
* **Model Architecture:** ResNet-18 (modified for 64x64 input)
* **Dataset:** Tiny-ImageNet-200
* **Data Partitioning:** Non-IID Dirichlet Distribution ($\alpha = 0.5$)
* **Federated Strategy:** FedAvg
* **Clients:** 10 active clients
* **Communication Rounds:** 50
* **Hyperparameters:** $lr=0.01$, $momentum=0.9$, $batch\_size=32$

### Results Summary
The model demonstrated stable convergence across 50 rounds, achieving a significant performance milestone on the 200-class classification task.

| Metric | Value |
| :--- | :--- |
| **Max Global Validation Accuracy** | **35.95%** |
| **Final Training Accuracy** | 59.97% |
| **Final Training Loss** | 1.4682 |
| **Convergence Round** | ~45 (plateau entry) |

### Visualizations
<p align="center">
  <img src="tiny_imagenet_accuracy.png" width="45%" />
  <img src="tiny_imagenet_loss.png" width="45%" />
</p>

---
