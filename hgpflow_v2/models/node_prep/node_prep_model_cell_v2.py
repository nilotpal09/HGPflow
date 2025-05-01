import torch
import torch.nn as nn
from ..helpers.dense import Dense
from ..helpers.diffusion_transformer import DiTEncoder
from ..helpers.graph_operations import custom_update_all, custom_copy_u, custom_sum_mailbox

class NodePrepModelCellV2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.track_init_net = Dense(**config['track_init_net'])

        self.calo_reg_emb_net = nn.Embedding(
            self.config['n_calo_reg'], self.config['calo_reg_emb_dim'])
        self.cell_init_net = Dense(**config['cell_init_net'])

        if 'node_transformer' in config.keys():
            self.node_transformer = DiTEncoder(**config['node_transformer'])


    def forward(self, batch):
        track_feat = self.track_init_net(batch['track']['feat0'])

        cell_calo_reg_emb = self.calo_reg_emb_net(batch['cell']['calo_region'].squeeze(-1))
        cell_feat0  = torch.cat([
            batch['cell']['feat0'], cell_calo_reg_emb], dim=-1)
        cell_feat  = self.cell_init_net(cell_feat0)

        track_mask = batch['pre_node']['is_track'].unsqueeze(-1)
        cell_mask = batch['pre_node']['is_cell'].unsqueeze(-1)

        pre_node_feat = track_feat * track_mask + cell_feat * cell_mask

        # # do we use this?
        # pre_node_mask = batch['pre_node']['mask']
        # pre_node_global = pre_node_feat.sum(dim=1) / \
        #         pre_node_mask.sum(dim=1, keepdim=True)

        # sum cells to get topo feat (tracks are just transferred)
        n_nodes = batch['node']['is_track'].size(1)
        prenode_to_node_mask = batch['prenode_to_node_edge_mask'] # B, N, M

        node_feat = custom_update_all(custom_copy_u, custom_sum_mailbox, 
            efn_src_feat      = pre_node_feat, 
            efn_num_nodes_dst = n_nodes,
            efn_src_mask      = batch['pre_node']['mask'], # (B, N)
            efn_dst_mask      = None, 
            efn_edge_mask     = prenode_to_node_mask
        )

        # do we use this?
        node_feat_global = node_feat.mean(dim=1) # there is no padding

        # identify what's track and what's topo
        track_mask = batch['node']['is_track'].unsqueeze(-1)
        topo_mask = batch['node']['is_topo'].unsqueeze(-1)
        node_feat = torch.cat([
            node_feat, track_mask.float(), topo_mask.float()], dim=-1)

        node_feat = self.node_transformer(q=node_feat, context=node_feat_global)

        node_feat = torch.cat(
            [node_feat, batch['node']['skip_feat0']], dim=-1)

        return node_feat