import torch.nn as nn
from .node_prep_model_mini import NodePrepModelMini
from .node_prep_model_cell_v1 import NodePrepModelCellV1
from .node_prep_model_cell_v2 import NodePrepModelCellV2


class NodePrepModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        model_dict = {
            'cell_v1': NodePrepModelCellV1,
            'cell_v2': NodePrepModelCellV2,
            'mini': NodePrepModelMini
        }

        self.model = model_dict[config['type']](config)

    def forward(self, batch):
        return self.model(batch)