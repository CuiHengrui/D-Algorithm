import copy
import math
import logging
import warnings
import pickle
import random
import sys
import time
from types import SimpleNamespace
import numpy as np
from plato.clients import simple
from plato.config import Config
from plato.processors.base import Processor
from plato.processors import model_quantize_qsgd
import torch

from multiple_processor import MultipleProcessor
import model_n_quantize

M = 1024 * 1024

# 获取装饰器（如果可用）
if hasattr(torch.utils.hooks, 'unserializable_hook'):
    unserializable_hook = torch.utils.hooks.unserializable_hook
else:
    # 如果不可用，创建一个空装饰器
    unserializable_hook = lambda x: x


class Client(simple.Client):
    """
    client节点会有进行两种方式的通信压缩(无压缩和QSDG)
    然后把两种量化结果进行上传

    上传:
        report: [super, data_size, cost_time]
        weight: [raw_weight, QSGD_weight]
    """

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):  
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", 
                               message="backward hook .* on tensor will not be serialized",
                               category=UserWarning)
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        self.config: Config = Config()
        self.processor: MultipleProcessor = []
        self.baseline_weights = {}

        self.random = random.Random(time.time_ns)
        client_config = Config().clients._asdict()

        # 样本大小 MB
        self.sample_size = client_config["sample_size"] * M

        # __init__函数只会调用一次初始化数组
        self.bandwidth_arr = client_config["bandwidth"]
        self.cpu_freq__arr = client_config["cpu_freq"]
        self.cn_arr = client_config["cn"]
        self.up_speed = 0
        self.cpu_freq = 0
        self.cn = 0
        self.freq_cost_sample = 0

    def init_processor(self, outbound_processor: MultipleProcessor):
        # 目前默认使用全精度和QSGD方法，之后可以添加其他方法
        for bit in (2, 4, 8, 16, 32):
            self.processor.add_processor(str(bit), model_n_quantize.Processor(n=bit))

    def configure(self) -> None:
        super().configure()
        self.processor = MultipleProcessor()
        self.init_processor(self.processor)
        # MB/s
        self.up_speed = self.bandwidth_arr[self.client_id - 1]
        # MHZ
        self.cpu_freq = self.cpu_freq__arr[self.client_id - 1]
        self.cn = self.cn_arr[self.client_id - 1]
        self.freq_cost_sample = self.sample_size * self.cn

        # MB
        self.model_size = (
            sys.getsizeof(pickle.dumps(self.trainer.model.cpu().state_dict())) / M
        )

    def create_gradient_hook(self, name):
        """创建可以序列化的梯度hook"""
        @unserializable_hook
        def hook(grad):
            self._gradient_hook(grad, name)
        return hook

    async def _train(self):
        """重写的训练方法，添加梯度收集和量化处理（避免hook方案）"""
        # 1. 保存初始模型状态
        self.baseline_weights = copy.deepcopy(self.trainer.model.state_dict())
    
        # 2. 运行原始训练过程
        report, updated_weights = await super()._train()

        # 3. 收集所有参数梯度
        all_gradients = []
        for name, param in self.trainer.model.named_parameters():
            if param.grad is not None:
                # 展平梯度并转换为列表
                flat_grad = param.grad.detach().cpu().view(-1)
                # 添加保护，避免梯度为空的参数
                if len(flat_grad) > 0:
                    # 随机采样最多10个梯度值
                    indices = torch.randperm(len(flat_grad))[:min(10, len(flat_grad))]
                    sampled_grad = flat_grad[indices].tolist()
                    all_gradients.extend(sampled_grad)
    
        # 4. 保存采样后的梯度分量
        if all_gradients:
            report.grad_components = random.sample(all_gradients, min(1000, len(all_gradients)))
        else:
            report.grad_components = [0.0]  # 防止空值
    
        # 5. 计算权重变化（delta）
        delta_weight = {}
        for name, current_weight in updated_weights.items():
            baseline = self.baseline_weights.get(name)
            if baseline is not None:
                delta_weight[name] = current_weight - baseline
            else:
                # 处理没有基准权重的特殊情况
                logging.warning(f"Baseline weight not found for {name}")
                delta_weight[name] = current_weight.clone()  # 直接使用当前权重
    
        # 6. 处理量化
        if hasattr(self, 'processor') and self.processor:
            self.processor.register(report)
            processed_weight = self.processor.process(delta_weight)
        else:
            processed_weight = delta_weight
    
        return report, processed_weight

    # def _gradient_hook(self, grad, name):
    #     """梯度hook，用于收集梯度分量"""
    #     if grad is not None:
    #         # 展平梯度并添加到列表中
    #         flat_grad = grad.detach().cpu().clone().view(-1).tolist()
    #         self.grad_components.extend(flat_grad)
    # # === 新增结束 ===
         
    def calcu_delta_weight(self, weight) -> dict[str, torch.Tensor]:
        deltas = {}
        for name, current_weight in weight.items():
            baseline = self.baseline_weights[name]

            # Calculate update
            _delta = current_weight - baseline
            deltas[name] = _delta
        return deltas

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Wrap up generating the report with any additional information."""
        train_squared_loss_step = self.trainer.run_history.get_metric_values(
            "train_squared_loss_step"
        )

        report.statistical_utility = report.num_samples * np.sqrt(
            1.0 / report.num_samples * sum(train_squared_loss_step)
        )

        report.t_compute = (self.freq_cost_sample * report.num_samples) / self.cpu_freq / 10 ** 9 / 200

        report.each_bit_time = self.model_size / 32 / self.up_speed

        report.compute_cost = (
            2
            * (10 ** (-28))
            / 2
            * ((self.cpu_freq) ** 2)
            * (self.freq_cost_sample * report.num_samples)
        )

        report.bandwidth = self.up_speed

        # === 新增：梯度统计信息 ===
        if hasattr(report, 'grad_components'):

        # 计算所有梯度分量的统计信息
            grad_array = np.array(report.grad_components)
            report.grad_min = float(np.min(grad_array))
            report.grad_max = float(np.max(grad_array))
        else:
            report.grad_min = 0.0
            report.grad_max = 0.0

        return report