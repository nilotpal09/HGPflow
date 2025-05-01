import sys
paths = sys.path
for p in paths:
     if '.local' in p:
             paths.remove(p)

import os
import yaml
import torch
from tqdm import tqdm
from pathlib import Path

from ..dataset.dataset import get_dataloader
from ..lightnings.hgpf_lightning import HGPFLightning
from ..utility.var_transformation import VarTransformation
from ..utility.tree_writer import TreeWriter
from ..utility.helper_dicts import class_mass_dict


class HyperedgeDataPrepHelper:

    def __init__(self, init_config):
        self.config_path_v = init_config['model']['config_path_v']
        self.config_v = yaml.safe_load(open(self.config_path_v, 'r'))

        self.config_ms1 = yaml.safe_load(
            open(init_config['model']['config_path_ms1'], 'r'))

        self.config_t = yaml.safe_load(
            open(init_config['model']['config_path_t'], 'r'))

        self.gpu = init_config['gpu']
        self.chunk_size = init_config['chunk_size']
        self.batch_size = init_config['batch_size']
        self.num_workers = init_config['num_workers']
        self.ind_threshold_loose = init_config['ind_threshold_loose']

        self.checkpoint_path = init_config['model']['checkpoint_path']
        self.load_model(self.checkpoint_path)

        self.transform_dict = {}
        for k, v in self.config_v['transformation_dict'].items():
            self.transform_dict[k] = VarTransformation(v)

    
    def load_model(self, checkpoint_path):
        self.lightning_model = HGPFLightning(self.config_v, self.config_ms1, None, self.config_t)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.lightning_model.load_state_dict(checkpoint['state_dict'])
        self.lightning_model.eval()

        if torch.cuda.is_available():
            print('switching to gpu')
            self.lightning_model.net.cuda()
            self.lightning_model.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def get_dataloader(self, inf_dict):
        ds_kwargs = {
            'filename': inf_dict['seg_path'],
            'config_v': self.config_v,
            'reduce_ds': inf_dict['n_events'],
            'compute_incidence': True}

        sampler_kwargs = {
            'config_v': self.config_v,
            'batch_size': self.batch_size,
            'remove_idxs': True,
            'apply_cells_threshold': self.config_t.get('apply_cells_threshold', False),
            'n_cells_threshold': self.config_t.get('n_cells_threshold', 10_000)}

        loader_kwargs = {
            'num_workers': self.num_workers,
            'pin_memory': True}

        return get_dataloader(self.config_v['dataset_type'], ds_kwargs, sampler_kwargs, loader_kwargs)


    def reset_dict_to_write(self):
        self.n_entry_buffer = 0
        if not hasattr(self, 'dict_to_write'):
            self.output_branches = [
                'ch_proxy_kin', 'neut_proxy_kin', 'proxy_is_charged', 
                'e_t', 'inc_times_node_feat', 'proxy_em_frac', 'node_feat_sum', 'pred_ind',
                'truth_pt', 'truth_eta', 'truth_phi', 'truth_ke', 'truth_class', 'truth_is_charged', 'truth_ind',
                'truth_pt_raw', 'truth_eta_raw', 'truth_ke_raw', 'proxy_pt_raw', 'proxy_ke_raw', 'proxy_eta_raw', 'proxy_phi'
            ]
            self.dict_to_write = {}
        for var in self.output_branches:
            self.dict_to_write[var] = []


    def prep_output_filepath(self, inf_dict):
        if inf_dict.get('output_filepath', None) is None:
            output_dir = self.config_path_v.replace('config_v.yml', 'inference')
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            inf_dict['output_filepath'] = \
                os.path.join(output_dir, inf_dict.get('dir_flag', ''), 'stage1_inference_' + inf_dict['seg_path'].split('/')[-1])
        else:
            output_dir = os.path.dirname(inf_dict['output_filepath'])
            Path(output_dir).mkdir(parents=True, exist_ok=True)


    def to_device(self, batch):
        if torch.is_tensor(batch):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self.to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self.to_device(v) for v in batch]
        else:
            return batch


    def run_inference(self, inf_dict):
        loader = self.get_dataloader(inf_dict)
        self.reset_dict_to_write()
        self.prep_output_filepath(inf_dict)

        output_obj = TreeWriter(
            inf_dict['output_filepath'], 'event_tree', chunk_size=self.chunk_size, dtype_to_32=True)

        for batch in tqdm(loader):
            batch = self.to_device(batch)

            (pred_inc, pred_ind, pred_is_charged), \
            (proxy_kin, proxy_is_charged, proxy_em_frac, e_t, inc_times_node_feat, node_feat_sum) = \
                self.lightning_model.net.forward_pre_stage2(batch)
            
            # get hungarian matched indices
            _, _, indices = self.lightning_model.metrics.LAP_loss_single(
                (pred_inc, pred_ind, pred_is_charged),
                (batch['incidence_truth'], batch['indicator_truth'], batch['particle']['is_charged']),
                node_is_track=batch['node']['is_track'])

            # reordering
            ch_proxy_kin, neut_proxy_kin = proxy_kin

            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, 3)
            ch_proxy_kin = torch.gather(ch_proxy_kin, 1, indices_expanded)
            neut_proxy_kin = torch.gather(neut_proxy_kin, 1, indices_expanded)

            proxy_is_charged = torch.gather(proxy_is_charged, 1, indices)
            proxy_em_frac = torch.gather(proxy_em_frac, 1, indices)

            pred_ind = torch.gather(pred_ind, 1, indices)

            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, e_t.size(-1))
            e_t = torch.gather(e_t, 1, indices_expanded)

            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, inc_times_node_feat.size(-1))
            inc_times_node_feat = torch.gather(inc_times_node_feat, 1, indices_expanded)

            # truth_ke computation
            truth_e_raw = batch['particle']['e_raw']
            truth_mass = torch.zeros_like(truth_e_raw)
            for c in class_mass_dict.keys():
                truth_mass[batch['particle']['class'] == c] = class_mass_dict[c]
            truth_ke_raw = truth_e_raw - truth_mass
            truth_ke_raw[truth_ke_raw < 1e-6] = 1e-6
            truth_ke = self.transform_dict['e'].forward(truth_ke_raw)


            # fill the dictionary
            bs = batch['indicator_truth'].size(0)
            for bs_idx in range(bs):
                
                mask = pred_ind[bs_idx] > self.ind_threshold_loose

                self.dict_to_write['ch_proxy_kin'].append(ch_proxy_kin[bs_idx][mask].view(-1))
                self.dict_to_write['neut_proxy_kin'].append(neut_proxy_kin[bs_idx][mask].view(-1))
                self.dict_to_write['proxy_is_charged'].append(proxy_is_charged[bs_idx][mask])
                self.dict_to_write['e_t'].append(e_t[bs_idx][mask].view(-1))
                self.dict_to_write['inc_times_node_feat'].append(inc_times_node_feat[bs_idx][mask].view(-1))
                self.dict_to_write['proxy_em_frac'].append(proxy_em_frac[bs_idx][mask])
                self.dict_to_write['pred_ind'].append(pred_ind[bs_idx][mask])
                self.dict_to_write['node_feat_sum'].append(node_feat_sum[bs_idx])

                self.dict_to_write['truth_pt'].append(batch['particle']['pt'][bs_idx][mask])
                self.dict_to_write['truth_eta'].append(batch['particle']['eta'][bs_idx][mask])
                self.dict_to_write['truth_phi'].append(batch['particle']['phi'][bs_idx][mask])
                self.dict_to_write['truth_ke'].append(truth_ke[bs_idx][mask])
                self.dict_to_write['truth_class'].append(batch['particle']['class'][bs_idx][mask])
                self.dict_to_write['truth_is_charged'].append(batch['particle']['is_charged'][bs_idx][mask])
                self.dict_to_write['truth_ind'].append(batch['indicator_truth'][bs_idx][mask])

                self.dict_to_write['truth_pt_raw'].append(batch['particle']['pt_raw'][bs_idx][mask])
                self.dict_to_write['truth_eta_raw'].append(batch['particle']['eta_raw'][bs_idx][mask])
                self.dict_to_write['truth_ke_raw'].append(truth_ke_raw[bs_idx][mask])

                self.dict_to_write['proxy_pt_raw'].append(
                    self.transform_dict['pt'].inverse(ch_proxy_kin[bs_idx][mask][:, 0]))
                self.dict_to_write['proxy_ke_raw'].append(
                    self.transform_dict['e'].inverse(neut_proxy_kin[bs_idx][mask][:, 0]))
                self.dict_to_write['proxy_eta_raw'].append(
                    self.transform_dict['eta'].inverse(
                        ch_proxy_kin[bs_idx][mask][:, 1] * proxy_is_charged[bs_idx][mask] + \
                        neut_proxy_kin[bs_idx][mask][:, 1] * (~proxy_is_charged[bs_idx][mask])))
                self.dict_to_write['proxy_phi'].append(
                    ch_proxy_kin[bs_idx][mask][:, 2] * proxy_is_charged[bs_idx][mask] + \
                    neut_proxy_kin[bs_idx][mask][:, 2] * (~proxy_is_charged[bs_idx][mask]))

                for k, v in self.dict_to_write.items():
                    self.dict_to_write[k][-1] = v[-1].detach().cpu().numpy()

            # time to write
            self.n_entry_buffer += bs
            if self.n_entry_buffer > self.chunk_size:
                output_obj.data = self.dict_to_write
                output_obj.write()
                self.reset_dict_to_write()

        # writing the last chunk
        if self.n_entry_buffer < self.chunk_size and self.n_entry_buffer > 0:
            output_obj.data = self.dict_to_write
            output_obj.write()

        print("\nMerged file:")
        print(inf_dict['output_filepath'])
        output_obj.f['event_tree'].show()
        output_obj.close()

