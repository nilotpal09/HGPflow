import torch
import numpy as np
import gc
import warnings
from ..utility.var_transformation import VarTransformation 


def get_val_step_perf(model_type, config_v, config_ms1, config_ms2, config_t):
    val_step_perf_dict = {
        'sup_attn': ValStepPerfStage1,
        'iterative_refiner': ValStepPerfStage1,
        'hyperedge_model': ValStepPerfStage2
    }
    return val_step_perf_dict[model_type](
        config_v, config_ms1, config_ms2, config_t)



class ValStepPerfStage1:
    def __init__(self, config_v, config_ms1, config_ms2, config_t):
        self.conf_t = config_t
        self.inc_bins = torch.linspace(0, 1, 21)
        self.reset()
        self.event_idxs_to_display = config_t.get('event_idxs_to_display', None)
        if self.event_idxs_to_display is not None:
            self.event_idxs_to_display = np.array(self.event_idxs_to_display)
        self.n_eds_to_fill = config_t.get('n_eds_to_fill', 0)


    def reset(self):
        self.log_dicts = []
        self.hg_summaries = []
        self.dep_es = []
        self.eds = []
        self.done_filling_eds = False
        gc.collect()


    def append_to_log_dicts(self, log_dict):
        self.log_dicts.append(log_dict)


    def run_on_batch(self, pred_inc, pred_ind_logits, pred_is_charged, indices, batch):
        pred_ind = torch.sigmoid(pred_ind_logits)

        pred_inc_ordered = torch.gather(pred_inc, 1, indices.unsqueeze(-1).expand(-1, -1, pred_inc.size(-1)))
        pred_ind_ordered = torch.gather(pred_ind.squeeze(-1), 1, indices)
        pred_is_charged_ordered = torch.gather(pred_is_charged, 1, indices)

        self.update_hg_summaries(batch, pred_inc_ordered, pred_ind_ordered)
        self.update_dep_e_summaries(batch, pred_inc_ordered, pred_ind_ordered, pred_is_charged_ordered)
        self.update_eds_incidence(batch, pred_inc_ordered, pred_ind_ordered)
    

    def update_hg_summaries(self, batch, pred_inc_ordered, pred_ind_ordered):
        hg_summary_dict = {}
        topo_mask = ~batch['node']['is_track'].squeeze(-1).bool().unsqueeze(1)

        inc_diff_means = []; inc_diff_variances = []; inc_count = []
        for bin_i in range(len(self.inc_bins)-1):
            inc_mask = (batch['incidence_truth'] >= self.inc_bins[bin_i]) & (batch['incidence_truth'] < self.inc_bins[bin_i+1])
            inc_mask = topo_mask * inc_mask * (batch['indicator_truth'] == 1).unsqueeze(-1)
            _diff = pred_inc_ordered[inc_mask] - batch['incidence_truth'][inc_mask]

            inc_diff_means.append(_diff.mean().detach().cpu().numpy())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                inc_diff_variances.append( ((_diff.std())**2).detach().cpu().numpy() )
            inc_count.append(inc_mask.sum().detach().cpu().numpy())
        
        hg_summary_dict['inc_diff_means'] = inc_diff_means
        hg_summary_dict['inc_diff_variances'] = inc_diff_variances
        hg_summary_dict['inc_count'] = inc_count

        # indicator (missing_rate and fake_rate)
        neut_part_mask = batch['particle']['class'].squeeze(-1) > 2
        neut_ind_targets = batch['indicator_truth'][neut_part_mask]
        neut_ind_preds   = pred_ind_ordered[neut_part_mask]

        ind_thresholds = torch.linspace(0.05, 0.95, 19)
        mr_num = []; mr_den = []; fr_num = []; fr_den = []
        for th in ind_thresholds:
            mr_num.append( ((neut_ind_preds < th) * (neut_ind_targets > 0.5)).sum().detach().cpu().numpy() )
            mr_den.append( (neut_ind_targets > 0.5).sum().detach().cpu().numpy() )
            fr_num.append( ((neut_ind_preds > th) * (neut_ind_targets < 0.5)).sum().detach().cpu().numpy() )
            fr_den.append( (neut_ind_preds > th).sum().detach().cpu().numpy() )

        hg_summary_dict['mr_num'] = mr_num; hg_summary_dict['mr_den'] = mr_den
        hg_summary_dict['fr_num'] = fr_num; hg_summary_dict['fr_den'] = fr_den
        if len(self.hg_summaries) == 0:
            hg_summary_dict['ind_thresholds'] = ind_thresholds.detach().cpu().numpy()

        self.hg_summaries.append(hg_summary_dict)
    

    def update_dep_e_summaries(self, batch, pred_inc_ordered, pred_ind_ordered, pred_is_charged_ordered):
        topo_e_raw = batch['topo']['e_raw'].unsqueeze(1)
        truth_raw_dep_e = (batch['incidence_truth'] * topo_e_raw).sum(2)
        pred_raw_dep_e = (pred_inc_ordered * topo_e_raw).sum(2)
        dep_e_dict = {
            'truth_raw_dep_e' : truth_raw_dep_e.detach().cpu().numpy(),
            'pred_raw_dep_e'  : pred_raw_dep_e.detach().cpu().numpy(),
            'truth_ind'       : batch['indicator_truth'].detach().cpu().numpy(),
            'pred_ind'        : pred_ind_ordered.detach().cpu().numpy(),
            'truth_is_charged': batch['particle']['is_charged'].bool().detach().cpu().numpy(),
            'pred_is_charged' : pred_is_charged_ordered.bool().detach().cpu().numpy()
        }
        self.dep_es.append(dep_e_dict)


    def update_eds_incidence(self, batch, pred_inc_ordered, pred_ind_ordered):
        if self.done_filling_eds:
            return

        if self.event_idxs_to_display is not None:
            idx_in_this_batch_to_fill = \
                np.where(np.isin(batch['idx'], self.event_idxs_to_display))[0]
            if len(idx_in_this_batch_to_fill) + len(self.eds) >= len(self.event_idxs_to_display):
                self.done_filling_eds = True

        else:
            idx_in_this_batch_to_fill = np.arange(
                min(self.n_eds_to_fill - len(self.eds), batch['incidence_truth'].size(0)))
            if len(idx_in_this_batch_to_fill) + len(self.eds) >= self.n_eds_to_fill:
                self.done_filling_eds = True

        for idx in idx_in_this_batch_to_fill:
            ed_dict = {
                'incidence_truth' : batch['incidence_truth'][idx].detach().cpu().numpy(),
                'indicator_truth' : batch['indicator_truth'][idx].detach().cpu().numpy(),
                'incidence_pred'  : pred_inc_ordered[idx].detach().cpu().numpy(),
                'indicator_pred'  : pred_ind_ordered[idx].detach().cpu().numpy(),
                'node_is_track'   : batch['node']['is_track'][idx].detach().cpu().numpy(),
                'node_e_raw'      : batch['topo']['e_raw'][idx].detach().cpu().numpy(),
                'node_pt_raw'     : batch['track']['pt_raw'][idx].detach().cpu().numpy(),
                'idx'             : batch['idx'][idx],
            }
            self.eds.append(ed_dict)



class ValStepPerfStage2:
    def __init__(self, config_v, config_ms1, config_ms2, config_t):
        self.conf_t = config_t
        self.store_regression_results = False
        if config_ms2['hyperedge_model'].get('kin_nets', None) is not None:
            self.store_regression_results = True
        self.reset()

        self.transform_dicts = {}
        for k, v in config_v['transformation_dict'].items():
            self.transform_dicts[k] = VarTransformation(v)

    def reset(self):
        self.log_dicts = []
        self.hyperedge_dicts = []
        gc.collect()

    def append_to_log_dicts(self, log_dict):
        self.log_dicts.append(log_dict)

    def run_on_batch(self, pred_kin, pred_class_logits, batch):
        '''
            we lose the event perspective here. flattened over batch and particles
        '''
        particle_dict = {}
        for v in ['truth_pt_raw', 'truth_ke_raw', 'truth_class', 'truth_ind', 'pred_ind']:
            particle_dict[v] = batch[v].view(-1).detach().cpu().numpy()

        ch_pred_class_logits, neut_pred_class_logits = pred_class_logits
        pred_class = torch.argmax(ch_pred_class_logits, dim=-1) * batch['proxy_is_charged'] + \
            (torch.argmax(neut_pred_class_logits, dim=-1) + 3) * (~batch['proxy_is_charged'])
        particle_dict['pred_class'] = pred_class.view(-1).detach().cpu().numpy()

        if self.store_regression_results:
            for v in ['truth_eta_raw', 'truth_phi', 'truth_pt', 'truth_ke', 'truth_eta',
                    'proxy_pt_raw', 'proxy_ke_raw', 'proxy_eta_raw', 'proxy_phi']:
                particle_dict[v] = batch[v].view(-1).detach().cpu().numpy()
            
            ch_pred_kin, neut_pred_kin = pred_kin

            particle_dict['pred_pt'] = ch_pred_kin[..., 0].view(-1).detach().cpu().numpy()
            particle_dict['pred_ke'] = neut_pred_kin[..., 0].view(-1).detach().cpu().numpy()
            particle_dict['pred_eta'] = (ch_pred_kin[..., 1] * batch['proxy_is_charged'] + \
                neut_pred_kin[..., 1] * (~batch['proxy_is_charged'])).view(-1).detach().cpu().numpy()
            particle_dict['pred_phi'] = (ch_pred_kin[..., 2] * batch['proxy_is_charged'] + \
                neut_pred_kin[..., 2] * (~batch['proxy_is_charged'])).view(-1).detach().cpu().numpy()
            
            particle_dict['pred_pt_raw'] = self.transform_dicts['pt'].inverse(
                ch_pred_kin[..., 0]).view(-1).detach().cpu().numpy()
            particle_dict['pred_ke_raw'] = self.transform_dicts['e'].inverse(
                neut_pred_kin[..., 0]).view(-1).detach().cpu().numpy()
            
            particle_dict['pred_eta_raw'] = self.transform_dicts['eta'].inverse(
                ch_pred_kin[..., 1] * batch['proxy_is_charged'] + \
                neut_pred_kin[..., 1] * (~batch['proxy_is_charged'])).view(-1).detach().cpu().numpy()

            particle_dict['pred_phi'] = (ch_pred_kin[..., 2] * batch['proxy_is_charged'] + \
                neut_pred_kin[..., 2] * (~batch['proxy_is_charged'])).view(-1).detach().cpu().numpy()
        
            particle_dict['proxy_pt'] = self.transform_dicts['pt'].forward(
                batch['proxy_pt_raw']).view(-1).detach().cpu().numpy()
            particle_dict['proxy_ke'] = self.transform_dicts['e'].forward(
                batch['proxy_ke_raw']).view(-1).detach().cpu().numpy()
            particle_dict['proxy_eta'] = self.transform_dicts['eta'].forward(
                batch['proxy_eta_raw']).view(-1).detach().cpu().numpy()

        self.hyperedge_dicts.append(particle_dict)
