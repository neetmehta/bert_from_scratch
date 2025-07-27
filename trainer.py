import torch
import os
from torch.utils.data import DataLoader
from typing import Callable, Optional
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_epochs: int = 10,
        checkpoint_dir: str = "./checkpoints",
        save_best_only: bool = True,
        early_stopping_patience: Optional[int] = None,
        metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.early_stopping_patience = early_stopping_patience
        self.metric_fn = metric_fn
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for step, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            self.optimizer.zero_grad()

            outputs = self.model(batch)
            loss = outputs['loss']
            loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def eval_epoch(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):

                outputs = self.model(batch)
                loss = outputs['loss']
                val_loss += loss.item()

        avg_loss = val_loss / len(self.val_loader)

        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: float):
        filename = os.path.join(self.checkpoint_dir, f"model_epoch{epoch}_loss{val_loss:.4f}.pt")
        torch.save(self.model.state_dict(), filename)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")

            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")

            val_loss, metrics = self.eval_epoch()
            print(f"Val Loss: {val_loss:.4f}")
            if "metric" in metrics:
                print(f"Val Metric: {metrics['metric']:.4f}")

            if self.scheduler:
                self.scheduler.step(val_loss)

            if not self.save_best_only or val_loss < self.best_val_loss:
                self.save_checkpoint(epoch, val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1

            elif self.early_stopping_patience is not None:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.early_stopping_patience:
                    print("Early stopping triggered.")
                    break
