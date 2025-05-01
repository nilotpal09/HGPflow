import torch
import random
from ..utility import misc
from ..dataset.dataset import get_dataloader

# Miscellaneous
from numpy.random import default_rng
SEED = 123456
RNG = default_rng(SEED)


class HGPFLightningIR:
    def __init__(self, parent_lightning, config_v, config_ms1, config_ms2, config_t):        
        self.config_v = config_v
        self.config_ms1 = config_ms1
        self.config_ms2 = config_ms2 # this is None
        self.config_t = config_t
        self._parent = parent_lightning

        self.T_TOTAL = self.config_ms1['hg_model']['T_TOTAL']
        self.T_BPTT  = self.config_ms1['hg_model']['T_BPTT']
        self.N_BPTT  = self.config_ms1['hg_model']['N_BPTT']

        self._parent.automatic_optimization = False
        self.random_bptt = config_t.get('random_bptt', False)
        if not self.random_bptt:
            self.sampler = misc.IntegerPartitionSampler(self.T_TOTAL-self.T_BPTT*self.N_BPTT, self.N_BPTT, RNG)


    def train_dataloader(self):
        ds_kwargs = {
            'filename': self.config_t['path_train'],
            'config_v': self.config_v,
            'reduce_ds': self.config_t['reduce_ds_train'],
            'compute_incidence': True}
        
        sampler_kwargs = {
            'config_v': self.config_v,
            'batch_size': self.config_t['batchsize_train'],
            'remove_idxs': True,
            'apply_cells_threshold': self.config_t.get('apply_cells_threshold', False),
            'n_cells_threshold': self.config_t.get('n_cells_threshold', 10_000)}

        loader_kwargs = {
            'num_workers': self.config_t['num_workers'],
            'pin_memory': True}

        return get_dataloader(self.config_v['dataset_type'], ds_kwargs, sampler_kwargs, loader_kwargs)


    def val_dataloader(self):
        ds_kwargs = {
            'filename': self.config_t['path_val'],
            'config_v': self.config_v,
            'reduce_ds': self.config_t['reduce_ds_val'],
            'compute_incidence': True}

        sampler_kwargs = {
            'config_v': self.config_v,
            'batch_size': self.config_t['batchsize_val'],
            'remove_idxs': True,
            'apply_cells_threshold': self.config_t.get('apply_cells_threshold', False),
            'n_cells_threshold': self.config_t.get('n_cells_threshold', 10_000)}

        loader_kwargs = {
            'num_workers': self.config_t['num_workers'],
            'pin_memory': True}

        return get_dataloader(self.config_v['dataset_type'], ds_kwargs, sampler_kwargs, loader_kwargs)


    def get_t_backprops(self, last_only=False):
        if last_only:
            return [False] * (self.T_TOTAL - 1) + [True]

        if self.random_bptt:
            bptt_list = [False] * (self.T_TOTAL - self.T_BPTT) + [True] * (self.T_BPTT - 1)
            random.shuffle(bptt_list)
            bptt_list.append(True)
            return [bptt_list]

        else:
            t_pre = self.sampler()
            bptt_lists = []
            for t in t_pre:
                bptt_list = [False] * t + [True] * (self.T_BPTT)
                bptt_lists.append(bptt_list)
            return bptt_lists

    
    def training_step(self, batch, batch_idx):

        bs = batch['incidence_truth'].size(0)
        opt = self._parent.optimizers()
        opt.zero_grad()

        node_feat = self._parent.net.node_prep_model(batch)
        e_t, v_t, i_t, track_eye, ch_mask_from_tracks = self._parent.net.hg_model.model.get_initial(
            node_feat, batch['node']['is_track'].bool())

        if 'hg_model' in self.config_t['train_components']:
            bptt_lists = self.get_t_backprops()

            loss_per_upd = []; loss_comps = []
            for t, bptt_list in enumerate(bptt_lists):
                preds_list, (e_t, v_t, i_t) = self._parent.net.hg_model.model.refine(
                    node_feat, e_t, v_t, i_t, batch['node']['is_track'].squeeze(-1).bool(), 
                    track_eye, ch_mask_from_tracks, t_backprops=bptt_list)

                loss, loss_components, _ = self._parent.metrics.LAP_loss_multi(
                    preds_list,
                    (batch['incidence_truth'], batch['indicator_truth'], batch['particle']['is_charged']),
                    node_is_track=batch['node']['is_track'])

                self._parent.manual_backward(loss)
                if not self._parent.automatic_optimization:
                    torch.nn.utils.clip_grad_norm_(self._parent.net.parameters(), 1.0)

                opt.step()
                opt.zero_grad()

                loss_per_upd.append(loss.detach().cpu().numpy())
                loss_comps.append(loss_components)

                if t < len(bptt_lists)-1:
                    node_feat = node_feat.detach()
                    e_t, v_t, i_t = e_t.detach(), v_t.detach(), i_t.detach()

            if self._parent.comet_logger is not None and \
                    batch_idx % self.config_t.get('train_log_every_n_steps', 1) == 0:
                logs = {}
                for k in loss_comps[0].keys():
                    logs[f'train/{k}'] = sum([l[k] for l in loss_comps]) / len(loss_comps)
                logs['train/loss_to_optimize_on'] = sum(loss_per_upd)/len(loss_per_upd)
                logs['grad_2.0_norm_total'] =  max(self._parent.norms2store)

                # global_step is incremented by 2 in the lightning module, and it's already done (-2)
                self._parent.comet_logger.log_metrics(logs, step=self._parent.global_step//2 - 1)

                self._parent.norms2store = []

        else:
            raise NotImplementedError

    def validation_step(self, batch, batch_idx):

        loss_to_optimize_on = 0

        node_feat = self._parent.net.node_prep_model(batch)
        preds_list, _ = self._parent.net.hg_model(
            node_feat, batch['node']['is_track'].squeeze(-1).bool())

        refiner_loss, loss_components, indices = self._parent.metrics.LAP_loss_single(
            preds_list[-1][-1], 
            (batch['incidence_truth'], batch['indicator_truth'], batch['particle']['is_charged']),
            node_is_track=batch['node']['is_track'])

        if 'hg_model' in self.config_t['train_components']:
            loss_to_optimize_on += refiner_loss.item()
            log_dict = {}
            for k, v in loss_components.items():
                log_dict[f'val/{k}'] = v

            inc_pred, ind_pred_logit, pred_is_charged = preds_list[-1][-1]

            self._parent.validation_step_perf.run_on_batch(
                inc_pred, ind_pred_logit, pred_is_charged, indices, batch)
                
        if 'kinematics' in self.config_t['train_components']:
            raise NotImplementedError
        
        log_dict['val_total_loss'] = loss_to_optimize_on
        self._parent.validation_step_perf.append_to_log_dicts(log_dict)
