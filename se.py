import copy
import logging
import math
import os
import pprint
from plato.servers import fedavg
from plato.config import Config
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

from multiple_processor import MultipleProcessor
from types import SimpleNamespace


# === SAC强化学习组件 ===
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

    def sample_action(self, state):
        with torch.no_grad():
            probs = self.forward(state)
            return torch.multinomial(probs, 1).item()


class SACAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005, lr=3e-4):
        self.q_net1 = QNetwork(state_dim, action_dim)
        self.q_net2 = QNetwork(state_dim, action_dim)
        self.target_q_net1 = QNetwork(state_dim, action_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim)

        # 初始化目标网络
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # 优化器
        self.q_optimizer = optim.Adam(
            list(self.q_net1.parameters()) + list(self.q_net2.parameters()), lr=lr
        )
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # 超参数
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.beta = 0.7  # 目标函数中的β参数
        self.min_range = -10.0  # 量化范围最小值
        self.max_range = 10.0  # 量化范围最大值

    def update(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        # 更新Q函数
        with torch.no_grad():
            # 计算目标Q值
            next_actions_probs = self.policy_net(next_states)
            next_q1 = self.target_q_net1(next_states, next_actions_probs)
            next_q2 = self.target_q_net2(next_states, next_actions_probs)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + self.gamma * (1 - dones) * next_q

        actions_onehot = F.one_hot(actions.squeeze(1), num_classes=self.action_dim).float()
        current_q1 = self.q_net1(states, actions_onehot)
        current_q2 = self.q_net2(states, actions_onehot)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # 更新策略网络
        actions_probs = self.policy_net(states)
        q1 = self.q_net1(states, actions_probs)
        q2 = self.q_net2(states, actions_probs)
        min_q = torch.min(q1, q2)

        # 添加熵正则化
        log_probs = torch.log(actions_probs + 1e-8)
        entropy = -torch.sum(actions_probs * log_probs, dim=1, keepdim=True)
        policy_loss = (-min_q - 0.2 * entropy).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 更新目标网络
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q_loss.item(), policy_loss.item()


# =========================================================================

class Server(fedavg.Server):
    def __init__(
            self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        self.processor: MultipleProcessor = MultipleProcessor()
        self.config: Config = Config()
        self.alpha = self.config.algorithm.alpha
        self.B = self.config.algorithm.B

        os.makedirs("./results/cost", exist_ok=True)
        self.record_file = f"./results/cost/{os.getpid()}.csv"
        with open(self.record_file, "w") as f:
            print(
                "round,total_time,compute_time,communication_time,communication_cost,", file=f
            )

        # === 添加强化学习组件 ===
        self.strategies = ['2', '4', '8', '16', '32']
        self.action_dim = len(self.strategies)

        # 状态维度: [带宽, 梯度最大值, 梯度最小值, 当前轮次]
        self.state_dim = 4
        self.agent = SACAgent(self.state_dim, self.action_dim)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 32

    def configure(self) -> None:
        return super().configure()

    def LQM(self, reports: list[SimpleNamespace]) -> list[str]:
        """使用强化学习生成LQM结果的筛选结果"""
        flags = []
        states = []
        actions = []

        # 1.为每个客户端选择策略
        for i, report in enumerate(reports):
            # 获取当前状态
            state = self.get_state(report)
            states.append(state)

            # 使用强化学习智能体选择策略
            action = self.agent.policy_net.sample_action(torch.FloatTensor(state).unsqueeze(0))
            actions.append(action)

            # 记录选择的策略
            strategy = self.strategies[action]
            flags.append(strategy)

        # 2. 计算奖励（目标函数g的最小化）
        for i, report in enumerate(reports):
            # 获取选定的量化策略
            selected_strategy = flags[i]
            bits_num = int(selected_strategy)

            # === 使用公式g计算目标函数值 ===
            # 第一部分: ∑(g_{i,v}^n - g_{i,v}^{-1})^2 / [0.4*(2^{bit_i}-1)^2]
            squared_sum = 0
            grad_components = report.grad_components
            min_val, max_val = report.grad_min, report.grad_max

            # 计算每个梯度分量的量化误差
            if min_val != max_val:
                for grad_value in grad_components:
                    error = self.quantization_error(grad_value, bits_num, min_val, max_val)
                    squared_sum += error ** 2
            else:
                # 如果所有梯度分量相同，平方和为零
                squared_sum = 0

            denominator = 0.4 * ((2 ** bits_num) - 1) ** 2
            part1 = squared_sum / (denominator + 1e-8)  # 防止除以0

            # 第二部分: β * (bits_num / bandwidth)
            bandwidth = report.bandwidth
            part2 = bits_num / bandwidth

            # 计算整体目标函数值
            q_value = self.agent.beta * part1 + (1 -self.agent.beta) * part2

            # 由于我们希望最小化q_value，因此奖励为负值
            reward = -q_value

            # 添加到经验回放缓冲区
            # 下一状态可以简单设为当前状态（因为状态转移不明确）
            self.replay_buffer.add(states[i], actions[i], reward, states[i], False)

        # 3. 如果缓冲区足够大，则更新强化学习模型
        if len(self.replay_buffer) > self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)
            self.agent.update(batch)

        logging.info(f"round {self.current_round} flags {flags}")
        return flags

    def quantization_error(self, value, bits, min_val, max_val):
        """计算量化误差"""
        if bits == 32:
            return 0.0  # 32位量化无误差

        # 计算量化步长
        num_levels = 2 ** bits - 1
        scale = (max_val - min_val) / num_levels
        if scale == 0:
            return 0.0

        # 模拟量化过程
        quantized = round((value - min_val) / scale) * scale + min_val

        return value - quantized

    def get_state(self, report: SimpleNamespace) -> np.ndarray:
        """获取强化学习状态"""
        # 状态包括: [带宽归一化, 梯度最大值, 梯度最小值, 当前轮次归一化]
        bandwidth = report.bandwidth
        max_g = report.grad_max
        min_g = report.grad_min
        round_normalized = self.current_round / self.config.trainer.rounds

        # 归一化处理
        bandwidth_norm = bandwidth / self.B
        max_g_norm = (max_g - self.agent.min_range) / (self.agent.max_range - self.agent.min_range + 1e-8)
        min_g_norm = (min_g - self.agent.min_range) / (self.agent.max_range - self.agent.min_range + 1e-8)

        return np.array([bandwidth_norm, max_g_norm, min_g_norm, round_normalized])

    def weights_received(self, deltas_received):
        reports = [update.report for update in self.updates]

        # 在报告中添加统计信息（如果需要）
        for report in reports:
            if not hasattr(report, 'grad_components'):
                report.grad_components = []
            if not hasattr(report, 'grad_min'):
                report.grad_min = 0.0
            if not hasattr(report, 'grad_max'):
                report.grad_max = 0.0

        # 目前使用client的loss作为statistical_utility，不需要计算范数
        for i, report in enumerate(reports):
            report.statistical_utility = {
                key: report.statistical_utility for key in deltas_received[i]
            }

        flags = self.LQM(reports)
        weights = []

        choosen_client_report = []
        for i, decompressed in enumerate(deltas_received):
            if flag := flags[i]:
                weights.append(decompressed[flag])
                reports[i].quantize_n = int(flag)
                choosen_client_report.append(reports[i])

        self.record(choosen_client_report)
        return super().weights_received(weights)

    def calcu_norm(self, deltas) -> int:
        norm = 0
        for _, delta in deltas.items():
            norm += torch.norm(delta)
        return norm

    def record(self, reports):
        # 记录总时间、计算开销（论文里的）、通信开销（上传的梯度总大小
        total_time = 0  # 总时间
        # 计算开销
        compute_time = np.array(list(map(lambda x: x.t_compute, reports)))
        compute_time_total = compute_time.sum()

        upload_time = np.array(
            list(map(lambda x: x.quantize_n * x.each_bit_time, reports))
        )
        communication_time = upload_time.sum()
        # compute_cost = sum(map(lambda x: x.compute_cost, reports))

        # print(list(map(lambda x: x.quantize_n * x.each_bit_time, reports)))

        total_time = max(compute_time + upload_time)
        # 通信开销
        communication_cost = (
            sum(map(lambda x: x.compressed_size[str(x.quantize_n)], reports))
        )

        with open(self.record_file, "a") as f:
            print(
                f"{self.current_round},{total_time},{compute_time_total},{communication_time},{communication_cost}",
                file=f,
            )