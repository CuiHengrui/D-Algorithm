from cl import Client
from distribution import FixNonIID
from ImageNet_k import DataSource
from se import Server
from algorithm import Algorithm
from trainer import Trainer

from plato.samplers.registry import registered_samplers
# from plato.datasources.registry import registered_datasources

from plato.samplers import distribution_noniid

registered_samplers["fixed_noniid"] = FixNonIID
# registered_datasources["ImageNet_1k"] = DataSource


def main():
    server = Server(algorithm=Algorithm)

    client = Client(trainer=Trainer)
    server.run(client=client)


if __name__ == "__main__":
    main()
