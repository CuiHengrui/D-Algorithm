from functools import partial
from torchvision.models.inception import Inception3

from cl import Client
from distribution import FixNonIID
from se import Server
from algorithm import Algorithm
from trainer import Trainer
from plato.config import Config

from plato.samplers.registry import registered_samplers
from CustomDataSource import CustomDataSource


registered_samplers["fixed_noniid"] = FixNonIID


def main():
    trainer = Trainer
    # 根据数据集和模型选择不同的参数配置
    if Config.data.datasource == "TinyImageNet":
        datasource = CustomDataSource
        if Config.trainer.model_name == "inception_v3":
            model = partial(Inception3, num_classes=Config.parameters.model.num_classes, aux_logits=True)
            client = Client(model=model, datasource=datasource, trainer=trainer)
            server = Server(algorithm=Algorithm, model=model, datasource=datasource, trainer=trainer)
        else:
            client = Client(trainer=trainer)
            server = Server(algorithm=Algorithm, trainer=trainer)
    else:
        datasource = None
        client = Client(trainer=trainer)
        server = Server(algorithm=Algorithm, trainer=trainer)
    server.run(client=client)


if __name__ == "__main__":
    main()
