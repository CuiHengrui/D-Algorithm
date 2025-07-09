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
            # 判断logits中是否有nan，如果有，抛出异常
            if torch.isnan(logits).any():
                raise ValueError("logits 中存在 NaN 值，训练终止。")
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

    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """
        针对 BN 层可能的报错：
        “ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 768, 1, 1])”
        重写 basie.Trainer 中的 get_train_loader 方法
        """
        return torch.utils.data.DataLoader(
            dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler, drop_last=True # 防止 BN 层报错
        )