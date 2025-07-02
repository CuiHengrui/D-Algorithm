from plato.algorithms import fedavg
from plato.trainers.base import Trainer


class Algorithm(fedavg.Algorithm):
    def __init__(self, trainer: Trainer):
        super().__init__(trainer)

    def compute_weight_deltas(self, baseline_weights, weights_received):
        # 上传已经为deltas
        return weights_received
