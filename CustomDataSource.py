
from plato.datasources import base
from plato.config import Config
from TinyImageNet import TinyImageNet
from torchvision import transforms

class CustomDataSource(base.DataSource):
    """
    自定义 TinyImageNet 数据源
    """
    def __init__(self):
        super().__init__()

        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

        self.trainset = TinyImageNet(root=Config().data.data_path, train=True, transform=transform)
        self.testset = TinyImageNet(root=Config().data.data_path, train=False, transform=transform)
    