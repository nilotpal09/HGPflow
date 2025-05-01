import torch
import torch.nn as nn

from .node_prep.node_prep_model import NodePrepModel
from .hg_learner.hg_learn_model import HGLearnModel
from .hyperedge_learner.hyperedge_model import HyperedgeModel
from .helpers.graph_operations import custom_update_all, custom_copy_u, custom_u_mul_e, custom_sum_mailbox
from ..utility.var_transformation import VarTransformation 

class HGPFlowModel(nn.Module):

    def __init__(self, config_v, config_ms1, config_ms2, class_mass_dict):
        super().__init__()

        self.epsilon = 1e-8
        self.config_ms1 = config_ms1
        self.config_ms2 = config_ms2
        self.class_mass_dict = class_mass_dict
        self.max_particles = config_v['max_particles']

        self.node_prep_model = NodePrepModel(config_ms1['node_prep_model'])
        self.hg_model = HGLearnModel(config_ms1['hg_model'], self.max_particles)
        if config_ms2 is not None:
            self.hyperedge_model = HyperedgeModel(config_ms2['hyperedge_model'])

        self.transform_funcs = {
            'pt' : VarTransformation(config_v['transformation_dict']['pt']),
            'e' : VarTransformation(config_v['transformation_dict']['e']),
            'eta' : VarTransformation(config_v['transformation_dict']['eta'])
        }


    def disable_gradients(self, component):
        if component == 'hg_learner':
            for p in self.node_prep_model.parameters():
                p.requires_grad = False
            for p in self.hg_model.parameters():
                p.requires_grad = False
        elif component == 'hyperedge_learner':
            for p in self.hyperedge_model.parameters():
                p.requires_grad = False
        else:
            raise ValueError(f'Component {component} not recognized')
        
    
    def infer(self, batch):
        with torch.no_grad():
            (pred_inc, pred_ind, pred_is_charged), \
            proxy_kin, pred_kin, pred_class_logits = self.forward(batch)

            ch_pred_class = torch.argmax(pred_class_logits[0], dim=-1)
            neut_pred_class = torch.argmax(pred_class_logits[1], dim=-1) + 3
            pred_class = ch_pred_class * pred_is_charged + neut_pred_class * (~pred_is_charged)

            proxy_ptetaphi_raw = self.get_ptetaphi_raw_from_kin(proxy_kin, pred_class, unnormalize=True)
            pred_ptetaphi_raw = self.get_ptetaphi_raw_from_kin(pred_kin, pred_class, unnormalize=True)

            return (pred_inc, pred_ind, pred_is_charged), proxy_ptetaphi_raw, pred_ptetaphi_raw, pred_class


    def forward(self, batch):
        (pred_inc, pred_ind, pred_is_charged), \
        (proxy_kin, proxy_is_charged, proxy_em_frac, e_t, inc_times_node_feat, node_feat_sum) = \
            self.forward_pre_stage2(batch)
        
        pred_kin, pred_class_logits = self.hyperedge_model(
            proxy_kin, proxy_is_charged, e_t, inc_times_node_feat, proxy_em_frac, node_feat_sum, pred_ind)

        return (pred_inc, pred_ind, pred_is_charged), proxy_kin, pred_kin, pred_class_logits


    def forward_pre_stage2(self, batch):
        node_feat = self.node_prep_model(batch)

        preds_list, (e_t, _, _) = self.hg_model(node_feat, batch['node']['is_track'])
        pred_inc, pred_ind_logit, pred_is_charged = preds_list[-1][1]
        pred_ind = torch.sigmoid(pred_ind_logit)

        proxy_kin, proxy_is_charged, proxy_em_frac = self.compute_proxies(batch, pred_inc)
        
        # (b, n_hyperedge, n_node) * (b, n_node, d_hid) -> (b, n_hyperedge, d_hid)
        inc_times_node_feat = torch.bmm(pred_inc, node_feat)        
        node_feat_sum = node_feat.sum(dim=1) 

        # proxy_is_charged is same as pred_is_charged (ToDo: remove redundant computation)
        return (pred_inc, pred_ind, pred_is_charged), \
            (proxy_kin, proxy_is_charged, proxy_em_frac, e_t, inc_times_node_feat, node_feat_sum)


    def compute_proxies(self, batch, inc):
        '''
        Args:
            inc: (B, n_hyperedge, n_node) (normalized and with track hard coded)
        Returns proxy (ch_pt, eta, phi, neut_e)
        '''
        bs, n_hedge, n_node = inc.size()

        inc_energy_raw = inc * batch['topo']['e_raw'].unsqueeze(-1).permute(0,2,1)

        inc_energy_raw = inc_energy_raw * batch['node']['is_topo'].unsqueeze(1) # kinda unnecessary
        inc = inc_energy_raw / (inc_energy_raw.sum(dim=2, keepdim=True) + 1e-8)

        # proxy computation for charged particles (track properties)
        track_eye = torch.eye(n_node, device=inc.device).unsqueeze(0).repeat(bs, 1, 1)
        track_eye = torch.nn.functional.pad(track_eye, (0, 0, 0, n_hedge - n_node))
        track_eye = track_eye * batch['node']['is_track'].unsqueeze(1) # kinda unnecessary

        proxy_is_charged = track_eye.sum(dim=2).bool() # B, n_hyperedge

        charged_proxy_pt = custom_update_all(custom_copy_u, custom_sum_mailbox, 
            efn_src_feat=batch['track']['pt'].unsqueeze(-1), efn_src_mask=batch['node']['is_track'],
            efn_num_nodes_dst=n_hedge, efn_dst_mask=None, efn_edge_mask=track_eye.permute(0,2,1))

        charged_proxy_eta = custom_update_all(custom_copy_u, custom_sum_mailbox, 
            efn_src_feat=batch['track']['eta'].unsqueeze(-1), efn_src_mask=batch['node']['is_track'],
            efn_num_nodes_dst=n_hedge, efn_dst_mask=None, efn_edge_mask=track_eye.permute(0,2,1))

        charged_proxy_phi = custom_update_all(custom_copy_u, custom_sum_mailbox, 
            efn_src_feat=batch['track']['phi'].unsqueeze(-1), efn_src_mask=batch['node']['is_track'],
            efn_num_nodes_dst=n_hedge, efn_dst_mask=None, efn_edge_mask=track_eye.permute(0,2,1))

        # charged pt, eta, phi
        charged_proxy_kin = torch.cat(
            [charged_proxy_pt, charged_proxy_eta, charged_proxy_phi], dim=-1) * proxy_is_charged.unsqueeze(-1)

        # proxy computation for neutral particles (incidence weighted sum)
        node_topo_mask = batch['node']['is_topo'].bool()
        node_topo_to_hedge_mask = batch['node']['is_topo'].unsqueeze(-1).repeat(1, 1, n_hedge).bool()

        neut_proxy_ke_raw = inc_energy_raw.sum(dim=2, keepdim=True) # B, n_hyperedge, 1

        neut_proxy_eta_raw = custom_update_all(custom_u_mul_e, custom_sum_mailbox, 
            efn_src_feat=batch['topo']['eta_raw'].unsqueeze(-1), efn_src_mask=node_topo_mask, 
            efn_num_nodes_dst=n_hedge, efn_dst_mask=None,
            efn_edge_feat=inc.permute(0,2,1).unsqueeze(-1), efn_edge_mask=node_topo_to_hedge_mask,
            nfn_edge_mask=node_topo_to_hedge_mask)
        
        neut_proxy_cosphi = custom_update_all(custom_u_mul_e, custom_sum_mailbox, 
            efn_src_feat=torch.cos(batch['topo']['phi']).unsqueeze(-1), efn_src_mask=node_topo_mask,
            efn_num_nodes_dst=n_hedge, efn_dst_mask=None,
            efn_edge_feat=inc.permute(0,2,1).unsqueeze(-1), efn_edge_mask=node_topo_to_hedge_mask,
            nfn_edge_mask=node_topo_to_hedge_mask)
        neut_proxy_sinphi = custom_update_all(custom_u_mul_e, custom_sum_mailbox,
            efn_src_feat=torch.sin(batch['topo']['phi']).unsqueeze(-1), efn_src_mask=node_topo_mask,
            efn_num_nodes_dst=n_hedge, efn_dst_mask=None, 
            efn_edge_feat=inc.permute(0,2,1).unsqueeze(-1), efn_edge_mask=node_topo_to_hedge_mask,
            nfn_edge_mask=node_topo_to_hedge_mask)
        neut_proxy_phi = torch.atan2(neut_proxy_sinphi, neut_proxy_cosphi)

        # apply transformation to neut proxy
        neut_proxy_ke  = self.transform_funcs['e'].forward(torch.clamp(neut_proxy_ke_raw, 0.0, None))
        neut_proxy_eta = self.transform_funcs['eta'].forward(neut_proxy_eta_raw)
        neut_proxy_kin = torch.cat(
            [neut_proxy_ke, neut_proxy_eta, neut_proxy_phi], dim=-1) * (~proxy_is_charged).unsqueeze(-1)

        # combined proxy
        proxy_kin = (charged_proxy_kin, neut_proxy_kin)

        # em_frac
        topo_em_fracs = batch['topo']['em_frac'].unsqueeze(1) # shape (b, 1, n_topo) # zero for tracks
        i_t_energy_em_raw = inc_energy_raw * topo_em_fracs # shape (b, n_hyperedge, n_topo)
        proxy_ke_em_raw = i_t_energy_em_raw.sum(dim=2)

        proxy_ke_raw = inc_energy_raw.sum(dim=2) # B, n_hyperedge, 1
        proxy_em_frac = proxy_ke_em_raw / (proxy_ke_raw + 1e-8)

        return proxy_kin, proxy_is_charged, proxy_em_frac


    def get_ptetaphi_raw_from_kin(self, kin, class_label, unnormalize=False):
        ch_kin, neut_kin = kin
        if unnormalize:           
            ch_pt_raw = self.transform_funcs['pt'].inverse(ch_kin[..., 0:1])
            ch_eta_raw = self.transform_funcs['eta'].inverse(ch_kin[..., 1:2])

            neut_ke_raw = self.transform_funcs['e'].inverse(neut_kin[..., 0:1])
            neut_eta_raw = self.transform_funcs['eta'].inverse(neut_kin[..., 1:2])
        else:
            ch_pt_raw = ch_kin[..., 0:1]
            ch_eta_raw = ch_kin[..., 1:2]

            neut_ke_raw = neut_kin[..., 0:1]
            neut_eta_raw = neut_kin[..., 1:2]

        ch_ptetaphi_raw = torch.cat([ch_pt_raw, ch_eta_raw, ch_kin[..., 2:3]], dim=-1)
        neut_pt_raw = neut_ke_raw / torch.cosh(neut_eta_raw) # photon (zero mass assumption)
        
        m_neut = self.class_mass_dict[3] # neutron mass in GeV
        nh_mask = class_label == 3
        neut_pt_raw[nh_mask] = torch.sqrt(
            torch.clamp(neut_ke_raw[nh_mask]**2 + 2*m_neut*neut_ke_raw[nh_mask], 0, None)) \
            / torch.cosh(neut_eta_raw[nh_mask])

        neut_ptetaphi_raw = torch.cat([neut_pt_raw, neut_eta_raw, neut_kin[..., 2:3]], dim=-1)

        ptetaphi_raw = ch_ptetaphi_raw * (class_label < 3).unsqueeze(-1) + \
            neut_ptetaphi_raw * (class_label >= 3).unsqueeze(-1)

        return ptetaphi_raw
