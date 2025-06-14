import torchvision 
from args import * 


class MNISTDataset(torchvision.datasets.MNIST):
    def __init__(self,args):
        transform = torchvision.transforms.Compose(
            [
                
                torchvision.transforms.Resize(args.image_size),
                torchvision.transforms.ToTensor(),
            ]
        )

        super().__init__(
            ".", train=True, download=True, transform=transform
        )

    def __getitem__(self, item):
        return super().__getitem__(item)[0]