import torch.nn as nn
from .iterative_refiner import IterativeRefiner
from .sup_attn import SupAttnModel


class HGLearnModel(nn.Module):

    def __init__(self, config, max_edges):
        super().__init__()
        self.config = config

        model_dict = {
            'iterative_refiner': IterativeRefiner,
            'sup_attn': SupAttnModel
        }
        self.model = model_dict[config['type']](config, max_edges)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)