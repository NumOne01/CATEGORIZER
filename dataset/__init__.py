from .cub import CUBirds
from .mit import MITs
from .dog import Dogs
from .air import Airs
from .car import Cars
from .cifar import Cifar100

from .import utils
from .base import BaseDataset


_type = {
    'cub': CUBirds,
    'mit': MITs,
    'dog': Dogs,
    'air': Airs,
    'cars': Cars,
    'cifar': Cifar100
}

def load(name, root, mode, transform=None):
    return _type[name](root=root, mode=mode, transform=transform)