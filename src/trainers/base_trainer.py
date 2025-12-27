import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseTrainer:
    def __init__(
        self,
        model: "BaseModel",
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: Optional[DataLoader],
        config: "Config",
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.criterion = nn.MSELoss()

        self.setup_logging()

    def setup_logging(self):
        self.exp_dir = Path(self.config.exp_dir) / self.config.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(self.exp_dir / "train.log"), logging.StreamHandler()],
    )

    def train(self):
        self.model.train()
        for epoch in range(self.config.epochs):
            self.optimizer.zero_grad()
            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch["labels"])
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
