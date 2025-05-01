import torch
import torch.nn as nn
from ..helpers.dense import Dense
from ..helpers.diffusion_transformer import DiTEncoder



class HyperedgeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.proxy_ch_kin_init_net = Dense(**config['proxy_ch_kin_init_net'])
        self.proxy_neut_kin_init_net = Dense(**config['proxy_neut_kin_init_net'])
        self.proxy_em_frac_init_net = Dense(**config['proxy_em_frac_init_net'])
        self.e_t_init_net = Dense(**config['e_t_init_net'])
        self.inc_times_node_feat_init_net = Dense(**config['inc_times_node_feat_init_net'])

        if 'transformer' in config:
            self.transformer = DiTEncoder(**config['transformer'])
            self.ind_threshold = config['ind_threshold']

        # classification is not optional
        self.ch_class_net = Dense(**config['class_nets']['ch_class_net'])
        self.neut_class_net = Dense(**config['class_nets']['neut_class_net'])

        # kinematics is optional
        if config.get('kin_nets', None) is not None:
            if 'ch_kin_net' in config['kin_nets']:
                self.ch_kin_net = Dense(**config['kin_nets']['ch_kin_net'])
            elif 'ch_pt_net' in config['kin_nets']:
                self.ch_pt_net = Dense(**config['kin_nets']['ch_pt_net'])

            if 'neut_kin_net' in config['kin_nets']:
                self.neut_kin_net = Dense(**config['kin_nets']['neut_kin_net'])
            elif 'neut_ke_net' in config['kin_nets']:
                self.neut_ke_net = Dense(**config['kin_nets']['neut_ke_net'])




    def get_part_init_feat(self, proxy_kin, proxy_is_charged, e_t, inc_times_node_feat, proxy_em_frac):
        ch_proxy_kin, neut_proxy_kin = proxy_kin

        # parsing pt, eta, phi, e
        proxy_pt = ch_proxy_kin[..., 0].unsqueeze(-1)
        proxy_ke = neut_proxy_kin[..., 0].unsqueeze(-1)

        proxy_eta = ch_proxy_kin[..., 1].unsqueeze(-1) * proxy_is_charged.unsqueeze(-1) + \
            neut_proxy_kin[..., 1].unsqueeze(-1) * (~proxy_is_charged.unsqueeze(-1))

        proxy_phi = ch_proxy_kin[..., 2].unsqueeze(-1) * proxy_is_charged.unsqueeze(-1) + \
            neut_proxy_kin[..., 2].unsqueeze(-1) * (~proxy_is_charged.unsqueeze(-1))
        proxy_cosphi = torch.cos(proxy_phi)
        proxy_sinphi = torch.sin(proxy_phi)

        # cat the inputs
        proxy_ch_kin_inp = torch.cat([
            proxy_pt, proxy_eta, proxy_cosphi, proxy_sinphi], dim=-1)

        proxy_neut_kin_inp = torch.cat([
            proxy_ke, proxy_eta, proxy_cosphi, proxy_sinphi], dim=-1)

        # init nets
        proxy_ch_kin_init   = self.proxy_ch_kin_init_net(proxy_ch_kin_inp)
        proxy_neut_kin_init = self.proxy_neut_kin_init_net(proxy_neut_kin_inp)
        proxy_kin_init = proxy_ch_kin_init * proxy_is_charged.unsqueeze(-1) + \
            proxy_neut_kin_init * (~proxy_is_charged.unsqueeze(-1))

        proxy_em_frac_init = self.proxy_em_frac_init_net(
            (proxy_em_frac.unsqueeze(-1) * 2) - 1  # [0,1] -> [-1,1]
        )
        e_t_init = self.e_t_init_net(e_t)
        i_t_times_node_feat_init = self.inc_times_node_feat_init_net(
            torch.clamp(inc_times_node_feat, -1, 1))

        # cat all the inits
        part_init_feat = torch.cat([
            proxy_kin_init, proxy_em_frac_init, e_t_init, i_t_times_node_feat_init
        ], dim=-1)

        return part_init_feat, (proxy_pt, proxy_ke, proxy_eta, proxy_phi)


    def forward(self, proxy_kin, proxy_is_charged, e_t, inc_times_node_feat, proxy_em_frac, node_feat_sum, ind):

        part_feat, (proxy_pt, proxy_ke, proxy_eta, proxy_phi) = \
            self.get_part_init_feat(
                proxy_kin, proxy_is_charged, e_t, inc_times_node_feat, proxy_em_frac)

        # transformer
        if hasattr(self, 'transformer'):
            not_part_mask = ind < self.ind_threshold
            part_feat = self.transformer(
                q=torch.cat([
                    part_feat, proxy_is_charged.unsqueeze(-1), ~proxy_is_charged.unsqueeze(-1)
                ], dim=-1),
                q_mask=not_part_mask, context=node_feat_sum
            )

        # regression (charged)
        if hasattr(self, 'ch_kin_net'):
            ch_del_kin = self.ch_kin_net(part_feat)

            ch_pred_pt = proxy_pt + ch_del_kin[..., 0:1]
            ch_pred_eta = proxy_eta + ch_del_kin[..., 1:2]
            ch_pred_phi = proxy_phi + ch_del_kin[..., 2:3]

            ch_pred_kin = torch.cat([ch_pred_pt, ch_pred_eta, ch_pred_phi], dim=-1)

        elif hasattr(self, 'ch_pt_net'):
            ch_del_pt = self.ch_pt_net(part_feat)
            ch_pred_pt = proxy_pt + ch_del_pt

            ch_pred_kin = torch.cat([ch_pred_pt, proxy_eta, proxy_phi], dim=-1)

        else:
            ch_pred_kin = torch.cat([proxy_pt, proxy_eta, proxy_phi], dim=-1)

        # regression (neutral)
        if hasattr(self, 'neut_kin_net'):
            neut_del_kin = self.neut_kin_net(part_feat)

            neut_pred_ke = proxy_ke + neut_del_kin[..., 0:1]
            neut_pred_eta = proxy_eta + neut_del_kin[..., 1:2]
            neut_pred_phi = proxy_phi + neut_del_kin[..., 2:3]

            neut_pred_kin = torch.cat([neut_pred_ke, neut_pred_eta, neut_pred_phi], dim=-1)

        elif hasattr(self, 'neut_ke_net'):
            neut_del_ke = self.neut_ke_net(part_feat)
            neut_pred_ke = proxy_ke + neut_del_ke

            neut_pred_kin = torch.cat([neut_pred_ke, proxy_eta, proxy_phi], dim=-1)

        else:
            neut_pred_kin = torch.cat([proxy_ke, proxy_eta, proxy_phi], dim=-1)

        # combine kinematics
        pred_kin = (ch_pred_kin, neut_pred_kin)

        # classification
        ch_class_logits = self.ch_class_net(part_feat)
        neut_class_logits = self.neut_class_net(part_feat)
        class_logits = (ch_class_logits, neut_class_logits)

        return pred_kin, class_logits