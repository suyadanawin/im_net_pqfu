from copy import deepcopy
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.models import get_model
from src.utils import accuracy_from_logits


class FLClient:
    def __init__(
        self,
        client_id: int,
        train_dataset,
        train_indices: List[int],
        config: dict,
        device: str
    ):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.train_indices = train_indices
        self.config = config
        self.device = device

        self.subset = Subset(self.train_dataset, self.train_indices)
        self.loader = DataLoader(
            self.subset,
            batch_size=self.config["fl"]["batch_size"],
            shuffle=True,
            num_workers=self.config["dataloader"]["num_workers"],
            pin_memory=self.config["dataloader"]["pin_memory"]
        )

    def train(self, global_model_state: Dict[str, torch.Tensor]):
        model = get_model(
            model_name=self.config["model"]["name"],
            num_classes=self.config["dataset"]["num_classes"],
            pretrained=self.config["model"]["pretrained"]
        )
        model.load_state_dict(deepcopy(global_model_state))
        model.to(self.device)
        model.train()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config["fl"]["lr"],
            momentum=self.config["fl"]["momentum"],
            weight_decay=self.config["fl"]["weight_decay"]
        )
        criterion = torch.nn.CrossEntropyLoss()

        local_epochs = self.config["fl"]["local_epochs"]

        epoch_losses = []
        epoch_accuracies = []

        for epoch in range(local_epochs):
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            pbar = tqdm(self.loader, desc=f"Client {self.client_id} Epoch {epoch+1}/{local_epochs}", leave=False)

            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                running_correct += accuracy_from_logits(outputs, labels, return_count=True)
                running_total += batch_size

                avg_loss = running_loss / running_total
                avg_acc = 100.0 * running_correct / running_total
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.2f}%")

            epoch_loss = running_loss / running_total
            epoch_acc = 100.0 * running_correct / running_total
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_acc)

        result = {
            "client_id": self.client_id,
            "num_samples": len(self.train_indices),
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "train_loss": float(sum(epoch_losses) / len(epoch_losses)),
            "train_acc": float(sum(epoch_accuracies) / len(epoch_accuracies)),
        }

        return result