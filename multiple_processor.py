from collections import OrderedDict
import copy
import math
import pickle
import sys
from typing import Any
from plato.processors.base import Processor
import time


M = 1024 ** 2
class MultipleProcessor(Processor):
    def __init__(self, name=None, trainer=None, **kwargs) -> None:
        super().__init__(name, trainer, **kwargs)
        self.processors: list[tuple[str, Processor]] = []
        self.report = None

    def add_processor(self, name: str, processor: Processor):
        self.processors.append((name, processor))

    def process(self, data: Any) -> Any:
        ret: dict[str, Any] = {}
        for name, processor in self.processors:
            start = time.process_time()
            ret[name] = processor.process(copy.deepcopy(data))
            end = time.process_time()
            if self.report:
                self.report.process_time[name] = end - start
                self.report.compressed_size[name] = sys.getsizeof(
                    pickle.dumps(data)
                # ) * math.log2(processor.n) / 32 / M
                ) * processor.n / 32 / M
        return ret

    def deprocess(self, data: Any):
        ret: dict[str, Any] = {}
        for name, processor in self.processors:
            start = time.process_time()
            ret[name] = processor.process(data[name])
            end = time.process_time()
            if self.report:
                self.report.process_time[name] = end - start
        return ret

    def register(self, report):
        self.report = report
        self.report.process_time = {}
        self.report.compressed_size = OrderedDict()





