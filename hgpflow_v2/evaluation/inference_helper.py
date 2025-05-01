import sys
paths = sys.path
for p in paths:
     if '.local' in p:
             paths.remove(p)

import os
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

from ..dataset.dataset import get_dataloader
from ..models.hgpflow_model import HGPFlowModel
from ..utility.var_transformation import VarTransformation
from ..utility.tree_writer import TreeWriter
from ..utility.helper_dicts import class_mass_dict


class InferenceHelper:

    def __init__(self, init_config):
        self.config_path_v = init_config['model']['config_path_v']
        self.config_v = yaml.safe_load(open(self.config_path_v, 'r'))

        self.config_ms1 = yaml.safe_load(open(init_config['model']['config_path_ms1'], 'r'))
        self.config_ms2 = yaml.safe_load(open(init_config['model']['config_path_ms2'], 'r'))

        self.gpu = init_config['gpu']
        self.chunk_size = init_config['chunk_size']
        self.batch_size = init_config['batch_size']
        self.num_workers = init_config['num_workers']
        self.ind_threshold_loose = init_config['ind_threshold_loose']

        self.apply_cells_threshold = init_config.get('apply_cells_threshold', False)
        self.n_cells_threshold = init_config.get('n_cells_threshold', 10_000)

        self.checkpoint_path = init_config['model']['checkpoint_path']
        self.load_model(self.checkpoint_path)

        self.transform_dicts = {}
        for k, v in self.config_v['transformation_dict'].items():
            self.transform_dicts[k] = VarTransformation(v)

    
    def load_model(self, checkpoint_path):
        self.hgpf_model = HGPFlowModel(self.config_v, self.config_ms1, self.config_ms2, class_mass_dict)
        
        state_dict = OrderedDict()
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[4:]] = v

        # # HACK
        # for k in list(state_dict.keys()):
        #     if k not in self.hgpf_model.state_dict():
        #         state_dict.pop(k)

        self.hgpf_model.load_state_dict(state_dict)
        self.hgpf_model.eval()

        if torch.cuda.is_available():
            print('switching to gpu')
            self.hgpf_model.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def get_dataloader(self, inf_dict):
        ds_kwargs = {
            'filename': inf_dict['seg_path'],
            'config_v': self.config_v,
            'reduce_ds': inf_dict['n_events'],
            'compute_incidence': False}

        sampler_kwargs = {
            'config_v': self.config_v,
            'batch_size': self.batch_size,
            'remove_idxs': False,
            'apply_cells_threshold': self.apply_cells_threshold,
            'n_cells_threshold': self.n_cells_threshold}

        loader_kwargs = {
            'num_workers': self.num_workers,
            'pin_memory': True}

        return get_dataloader(self.config_v['dataset_type'], ds_kwargs, sampler_kwargs, loader_kwargs)


    def prep_dict_to_write(self):
        n_events = len(self.sorted_event_numbers)
        self.dict_to_write = {}
        for part_type in ['truth', 'proxy', 'hgpflow']:
            tmp_dict = {}
            for var in ['pt', 'eta', 'phi', 'class']:
                tmp_dict[var] = [[] for _ in range(n_events)]
            self.dict_to_write[part_type] = tmp_dict
        self.dict_to_write['proxy'].pop('class')

        self.dict_to_write['pred_ind'] = [[] for _ in range(n_events)]
        self.dict_to_write['event_number'] = self.sorted_event_numbers


    def fill_dict_to_write(self, batch, proxy_ptetaphi_raw, pred_ptetaphi_raw, pred_class, pred_ind):
        event_numbers = batch['event_number']
        for i, event_number in enumerate(event_numbers):
            idx = np.where(self.sorted_event_numbers == event_number)[0][0]

            truth_ind_mask = batch['indicator_truth'][i] > 0.5

            self.dict_to_write['truth']['pt'][idx].extend(
                batch['particle']['pt_raw'][i][truth_ind_mask].tolist())
            self.dict_to_write['truth']['eta'][idx].extend(
                batch['particle']['eta_raw'][i][truth_ind_mask].tolist())
            self.dict_to_write['truth']['phi'][idx].extend(
                batch['particle']['phi'][i][truth_ind_mask].tolist())
            self.dict_to_write['truth']['class'][idx].extend(
                batch['particle']['class'][i][truth_ind_mask].tolist())

            pred_ind_mask = pred_ind[i] > self.ind_threshold_loose

            self.dict_to_write['proxy']['pt'][idx].extend(
                proxy_ptetaphi_raw[i, :, 0][pred_ind_mask].tolist())
            self.dict_to_write['proxy']['eta'][idx].extend(
                proxy_ptetaphi_raw[i, :, 1][pred_ind_mask].tolist())
            self.dict_to_write['proxy']['phi'][idx].extend(
                proxy_ptetaphi_raw[i, :, 2][pred_ind_mask].tolist())

            self.dict_to_write['hgpflow']['pt'][idx].extend(
                pred_ptetaphi_raw[i, :, 0][pred_ind_mask].tolist())
            self.dict_to_write['hgpflow']['eta'][idx].extend(
                pred_ptetaphi_raw[i, :, 1][pred_ind_mask].tolist())
            self.dict_to_write['hgpflow']['phi'][idx].extend(
                pred_ptetaphi_raw[i, :, 2][pred_ind_mask].tolist())
            self.dict_to_write['hgpflow']['class'][idx].extend(
                pred_class[i][pred_ind_mask].cpu().numpy())

            self.dict_to_write['pred_ind'][idx].extend(pred_ind[i][pred_ind_mask].tolist())


    def get_output_filepath(self, inf_dict):
        if inf_dict.get('pred_path', None) is None:
            output_dir = self.config_path_v.replace('config_v.yml', 'inference')
            output_filepath = \
                os.path.join(output_dir, inf_dict.get('dir_flag', ''), 'pred_' + inf_dict['seg_path'].split('/')[-1].replace('.root', '_merged.root'))
        else:
            output_filepath = inf_dict['output_filepath']
        return output_filepath


    def write_to_file(self, output_filepath):        
        merged_output_obj = TreeWriter(
            output_filepath, 'event_tree', chunk_size=self.chunk_size)
        merged_output_obj.write_dict_in_chunk(self.dict_to_write, desc='writing')

        print("\nMerged file:")
        print(output_filepath)
        merged_output_obj.f['event_tree'].show()
        merged_output_obj.close()


    def to_device(self, batch):
        if torch.is_tensor(batch):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self.to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self.to_device(v) for v in batch]
        else:
            return batch


    def run_prediction(self, inf_dict):
        loader = self.get_dataloader(inf_dict)

        # print in cyan
        if loader.dataset.n_particles.max() > loader.dataset.max_particles:
            print('\033[96m' + f'Warning! Setting max particles in the dataloader to {loader.dataset.n_particles.max()}' + '\033[0m')
            print('\033[96m' + 'Careful as some things may break (like truth incidence computation, hungarian matching etc)' + '\033[0m')
            loader.dataset.max_particles = loader.dataset.n_particles.max()

        unique_event_numbers = np.unique(loader.dataset.data_dict['event_number'])
        self.sorted_event_numbers = np.sort(unique_event_numbers)
        self.prep_dict_to_write()

        for batch in tqdm(loader):
            batch = self.to_device(batch)

            # running inference
            (_, pred_ind, _), proxy_ptetaphi_raw, pred_ptetaphi_raw, pred_class = self.hgpf_model.infer(batch)

            self.fill_dict_to_write(batch, proxy_ptetaphi_raw, pred_ptetaphi_raw, pred_class, pred_ind)

        pred_path = self.get_output_filepath(inf_dict)
        self.write_to_file(pred_path)
