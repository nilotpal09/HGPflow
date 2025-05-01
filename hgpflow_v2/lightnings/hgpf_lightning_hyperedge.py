import torch
import torch.nn.functional as F
from ..dataset.dataset import get_dataloader
from collections import OrderedDict
import gc


class HGPFLightningHyperedge:

    def __init__(self, parent_lightning, config_v, config_ms1, config_ms2, config_t):        
        self.config_v = config_v
        self.config_ms1 = config_ms1
        self.config_ms2 = config_ms2
        self.config_t = config_t
        self._parent = parent_lightning

        ckpt = torch.load(config_t['checkpoint_ms1'], map_location='cpu')

        node_prep_state_dict = OrderedDict()
        hg_model_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if 'node_prep_model' in k:
                node_prep_state_dict[k.replace('net.node_prep_model.', '')] = v
            elif 'hg_model' in k:
                hg_model_state_dict[k.replace('net.hg_model.', '')] = v

        self._parent.net.node_prep_model.load_state_dict(node_prep_state_dict)
        self._parent.net.hg_model.load_state_dict(hg_model_state_dict)

        self._parent.net.disable_gradients('hg_learner')

        self.loss_wts = config_t['loss_wts']
        self.class_based_wts = config_t['class_based_wts']


    def train_dataloader(self):
        ds_kwargs = {
            'filename': self.config_t['path_train'],
            'ind_threshold': self.config_ms2['hyperedge_model']['ind_threshold'],
            'reduce_ds': self.config_t['reduce_ds_train']}

        sampler_kwargs = None

        loader_kwargs = {
            'num_workers': self.config_t['num_workers'],
            'pin_memory': True, 'batch_size': self.config_t['batchsize_train']}

        return get_dataloader('hyperedge', ds_kwargs, sampler_kwargs, loader_kwargs)
    

    def val_dataloader(self):
        ds_kwargs = {
            'filename': self.config_t['path_val'],
            'ind_threshold': self.config_ms2['hyperedge_model']['ind_threshold'],
            'reduce_ds': self.config_t['reduce_ds_val']}

        sampler_kwargs = None

        loader_kwargs = {
            'num_workers': self.config_t['num_workers'],
            'pin_memory': True, 'batch_size': self.config_t['batchsize_val']}

        return get_dataloader('hyperedge', ds_kwargs, sampler_kwargs, loader_kwargs)


    def reg_loss(self, ch_or_neut, pred_kin, truth_kin, truth_class, truth_ind, pred_ind):
        pt_or_ke = 'pt' if ch_or_neut == 'ch' else 'ke'

        pt_or_e_loss = F.mse_loss(
            pred_kin[..., 0], truth_kin[0], reduction='none')
        eta_loss = F.mse_loss(
            pred_kin[..., 1], truth_kin[1], reduction='none')
        phi_loss = 1 - torch.cos(
            pred_kin[..., 2] - truth_kin[2])

        # class based weights
        class_based_wt = torch.ones_like(pt_or_e_loss)
        for cl_i, cl_wt in enumerate(self.class_based_wts[ch_or_neut]):
            class_based_wt[truth_class == cl_i] = cl_wt

        # indicator based weights # soft wt and a hard threshold
        f = pred_ind * (pred_ind > self.config_ms2['hyperedge_model']['ind_threshold']).float() * (truth_ind > 0.5).float()
        f_sum = f.sum() + 1e-8

        pt_or_e_loss = self.loss_wts[ch_or_neut][pt_or_ke] * (pt_or_e_loss * class_based_wt * f).sum() / f_sum
        eta_loss     = self.loss_wts[ch_or_neut]['eta'] * (eta_loss * class_based_wt * f).sum() / f_sum
        phi_loss     = self.loss_wts[ch_or_neut]['phi'] * (phi_loss * class_based_wt * f).sum() / f_sum

        reg_loss = pt_or_e_loss + eta_loss + phi_loss

        loss_components = {
            f'{ch_or_neut}_{pt_or_ke}_loss': pt_or_e_loss.item(),
            f'{ch_or_neut}_eta_loss': eta_loss.item(),
            f'{ch_or_neut}_phi_loss': phi_loss.item(),
            f'{ch_or_neut}_total_loss': reg_loss.item()}

        return reg_loss, loss_components


    def class_loss(self, pred_logits, truth_class, is_charged, truth_ind, pred_ind):
        IND_TH = self.config_ms2['hyperedge_model']['ind_threshold']

        ch_mask = is_charged * (truth_ind > 0.5)
        neut_mask = (~is_charged) * (truth_ind > 0.5)

        # class loss
        f_ch = pred_ind[ch_mask] * (pred_ind[ch_mask] > IND_TH)
        f_neut = pred_ind[neut_mask] * (pred_ind[neut_mask] > IND_TH)
        f_ch_sum = f_ch.sum() + 1e-8; f_neut_sum = f_neut.sum() + 1e-8

        truth_class_ch   = truth_class[ch_mask]
        truth_class_neut = truth_class[neut_mask] - 3

        pred_logits_ch, pred_logits_neut = pred_logits
        pred_logits_ch = pred_logits_ch[ch_mask]
        pred_logits_neut = pred_logits_neut[neut_mask]

        ch_class_loss  = F.cross_entropy(pred_logits_ch, truth_class_ch.long(), 
            reduction='none', weight=torch.tensor(self.class_based_wts['ch'], device=pred_logits_ch.device))
        ch_class_loss = self.loss_wts['ch']['class'] * (ch_class_loss * f_ch.view(-1)).sum() / f_ch_sum

        neut_class_loss  = F.cross_entropy(pred_logits_neut, truth_class_neut.long(),
            reduction='none', weight=torch.tensor(self.class_based_wts['neut'], device=pred_logits_neut.device))
        neut_class_loss = self.loss_wts['neut']['class'] * (neut_class_loss * f_neut.view(-1)).sum() / f_neut_sum
        
        class_loss = neut_class_loss + ch_class_loss
        loss_components = {
            'ch_class_loss': ch_class_loss.item(),
            'neut_class_loss': neut_class_loss.item(),
            'total_class_loss': class_loss.item()}

        return class_loss, loss_components
    
        
    def compute_loss(self, pred_kin, pred_class_logits, batch):
        loss_to_optimize_on = 0; loss_components = {}

        print('\nfrom compute_loss')
        print(pred_class_logits[0].shape, batch['truth_class'].shape, batch['truth_is_charged'].shape, batch['truth_ind'].shape, batch['pred_ind'].shape)

        # classification is a must
        class_loss, class_loss_components = self.class_loss(
            pred_class_logits, batch['truth_class'], 
            batch['truth_is_charged'], batch['truth_ind'], batch['pred_ind'])
        loss_to_optimize_on += class_loss
        for k, v in class_loss_components.items():
            loss_components[k] = v

        # regression is optional
        if self.config_ms2['hyperedge_model'].get('kin_nets', None) is not None:
            if 'ch_kin_net' in self.config_ms2['hyperedge_model']['kin_nets'] or \
                    'ch_pt_net' in self.config_ms2['hyperedge_model']['kin_nets']:
                ch_reg_loss, reg_loss_components = self.reg_loss('ch', pred_kin[0], 
                    (batch['truth_pt'], batch['truth_eta'], batch['truth_phi']), 
                    batch['truth_is_charged'], batch['truth_ind'], batch['pred_ind'])
                loss_to_optimize_on += ch_reg_loss
                for k, v in reg_loss_components.items():
                    loss_components[k] = v

            if 'neut_kin_net' in self.config_ms2['hyperedge_model']['kin_nets'] or \
                    'neut_ke_net' in self.config_ms2['hyperedge_model']['kin_nets']:
                neut_reg_loss, reg_loss_components = self.reg_loss('neut', pred_kin[1], 
                    (batch['truth_ke'], batch['truth_eta'], batch['truth_phi']), 
                    ~batch['truth_is_charged'], batch['truth_ind'], batch['pred_ind'])
                loss_to_optimize_on += neut_reg_loss
                for k, v in reg_loss_components.items():
                    loss_components[k] = v

        loss_components['loss_to_optimize_on'] = loss_to_optimize_on.item()

        return loss_to_optimize_on, loss_components


    def training_step(self, batch, batch_idx):
        pred_kin, class_logits = self._parent.net.hyperedge_model(
            (batch['ch_proxy_kin'], batch['neut_proxy_kin']), batch['proxy_is_charged'], batch['e_t'], 
            batch['inc_times_node_feat'], batch['proxy_em_frac'], batch['node_feat_sum'], batch['pred_ind']
        )
        
        loss, loss_components = self.compute_loss(pred_kin, class_logits, batch)
        if self._parent.comet_logger is not None and \
                batch_idx % self.config_t.get('train_log_every_n_steps', 1) == 0:
            logs = {f'train/{k}': v for k,v in loss_components.items()}
            logs['grad_2.0_norm_total'] =  max(self._parent.norms2store) if len(self._parent.norms2store) > 0 else 0
            self._parent.comet_logger.log_metrics(logs, step=self._parent.global_step - 1)

        return loss

    def validation_step(self, batch, batch_idx):
        pred_kin, class_logits = self._parent.net.hyperedge_model(
            (batch['ch_proxy_kin'], batch['neut_proxy_kin']), batch['proxy_is_charged'], batch['e_t'], 
            batch['inc_times_node_feat'], batch['proxy_em_frac'], batch['node_feat_sum'], batch['pred_ind']
        )
        
        loss, loss_components = self.compute_loss(pred_kin, class_logits, batch)
        
        log_dict = {f'val/{k}': v for k, v in loss_components.items()}
        log_dict['val_total_loss'] = loss.item()
        self._parent.validation_step_perf.append_to_log_dicts(log_dict)

        self._parent.validation_step_perf.run_on_batch(pred_kin, class_logits, batch)