import torch
import torch.nn as nn
from ..helpers.dense import Dense
from ..helpers.diffusion_transformer import DiTEncoder


class NodePrepModelMini(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.track_init_net = Dense(**config['track_init_net'])
        self.topo_init_net = Dense(**config['topo_init_net'])

        self.node_transformer = DiTEncoder(**config['transformer'])


    def forward(self, batch):
        track_feat = self.track_init_net(batch['track']['feat0'])
        topo_feat = self.topo_init_net(batch['topo']['feat0'])

        track_mask = batch['node']['is_track'].unsqueeze(-1)
        topo_mask = batch['node']['is_topo'].unsqueeze(-1)
        node_feat = track_feat * track_mask + topo_feat * topo_mask

        # transformer
        node_global = node_feat.mean(dim=1) # there is no padding

        # identify what's track and what's topo
        node_feat = torch.cat([
            node_feat, track_mask.float(), topo_mask.float()], dim=-1)

        node_feat = self.node_transformer(q=node_feat, context=node_global)

        if self.config.get('add_skip_feat', False):
            node_feat = torch.cat(
                [node_feat, batch['node']['skip_feat0']], dim=-1)

        return node_feat