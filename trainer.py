"""
基于oort的实现，计算训练时损失
"""

import numpy as np
import torch
from torch import nn
from plato.trainers import basic


class Trainer(basic.Trainer):
    """A federated learning trainer used by the Oort that keeps track of losses."""

    def process_loss(self, outputs, labels) -> torch.Tensor:
        """Returns the loss from CrossEntropyLoss, and records the sum of
        squaures over per_sample loss values."""
        loss_func = nn.CrossEntropyLoss(reduction="none")
        per_sample_loss = loss_func(outputs, labels)

        # Stores the sum of squares over per_sample loss values
        self.run_history.update_metric(
            "train_squared_loss_step",
            sum(np.power(per_sample_loss.cpu().detach().numpy(), 2)),
        )

        return torch.mean(per_sample_loss)

    def get_loss_criterion(self):
        """Returns the loss criterion."""
        return self.process_loss

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """
        重载父类方法，针对inception_v3修改
        """
        self.optimizer.zero_grad()
        outputs = self.model(examples)

        if config["model_name"] == "inception_v3":
            # 修改代码（针对inception_v3）
            logits, aux_logits = outputs.logits, outputs.aux_logits
            # 在原文中，总的损失为 1*主分类器损失 + 0.3*辅助分类器损失
            loss = self._loss_criterion(logits, labels) + 0.3 * self._loss_criterion(aux_logits, labels)
        else:
            # 原始代码
            loss = self._loss_criterion(outputs, labels)

        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.step()

        return loss