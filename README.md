## 📈 Phase 2: Federated Training Baseline
In this phase, we established a robust global baseline model trained under extreme data skew. This serves as the "Pre-unlearning" state for Phase 3.

<<<<<<< HEAD
### Experiment Setup
* **Model Architecture:** ResNet-18 (modified for 64x64 input)
* **Dataset:** Tiny-ImageNet-200
* **Data Partitioning:** Non-IID Dirichlet Distribution ($\alpha = 0.5$)
* **Federated Strategy:** FedAvg
* **Clients:** 10 active clients
* **Communication Rounds:** 50
* **Hyperparameters:** $lr=0.01$, $momentum=0.9$, $batch\_size=32$
=======
### 1. Environment & Model Architecture
* **Model:** `resnet18` (Adapted for 64x64 input, non-pretrained)
* **Dataset:** Tiny-ImageNet-200 (200 unique classes)
* **Hardware:** Local execution via NVIDIA GeForce RTX 3070
>>>>>>> 8bf996dd7 (Docs: Update Phase 2 visuals grid in README)

### 2. Federated Learning Parameters (Extreme Non-IID)
| Configuration | Value |
| :--- | :--- |
| **Total Clients** | 10 |
| **Data Distribution** | **Extreme Non-IID Dirichlet ($\alpha = 0.1$)** |
| **Communication Rounds** | 50 |
| **Batch Size** | 64 (Train) / 128 (Eval) |
| **Optimization** | SGD ($lr=0.01$, $momentum=0.9$, $weight\_decay=0.0005$) |

### 3. Baseline Performance Results
The global model demonstrated steady convergence despite the high data heterogeneity.

* **Max Global Validation Accuracy:** **35.95%**
* **Final Training Loss:** 1.4682
* **Final Training Accuracy:** 59.97%

#### **Phase 2 Federated Training Visualizations**

| Training Accuracy | Validation Accuracy |
| :---: | :---: |
| ![Client Train Accuracy](./client_train_accuracy.png) | ![Global Val Accuracy](./global_val_accuracy.png) |
| **Validation Loss** | **Train vs Val Accuracy** |
| ![Global Val Loss](./global_val_loss.png) | ![Train vs Val Accuracy](./train_vs_val_accuracy.png) |

---
