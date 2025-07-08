from functools import partial
from torchvision.models import inception_v3

from cl import Client
from distribution import FixNonIID
from se import Server
from algorithm import Algorithm
from trainer import Trainer
from plato.config import Config

from plato.samplers.registry import registered_samplers
# from plato.datasources.registry import registered_datasources
from CustomDataSource import CustomDataSource


from plato.samplers import distribution_noniid

registered_samplers["fixed_noniid"] = FixNonIID
# registered_datasources["ImageNet_1k"] = DataSource


def main():
    trainer = Trainer
    if Config.trainer.model_name == "inception_v3":
        datasource = CustomDataSource
        model = partial(inception_v3, num_classes=200, aux_logits=True)
        client = Client(model=model,datasource=datasource, trainer=trainer)
        server = Server(model=model, datasource=datasource, trainer=trainer) 
    else:
        client = Client(trainer=trainer)
        server = Server(algorithm=Algorithm, trainer=trainer)
    
    server.run(client=client)


if __name__ == "__main__":
    main()
