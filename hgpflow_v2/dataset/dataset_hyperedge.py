import uproot
import numpy as np
import awkward as ak
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import gc
import glob


class PflowDatasetHyperedge(Dataset):

    def __init__(self, filename, ind_threshold, reduce_ds=-1):
        super().__init__()
	    
        self.filename = filename
        self.ind_threshold = ind_threshold
        self.reduce_ds = reduce_ds
        assert reduce_ds == -1 or (isinstance(reduce_ds, int) and reduce_ds > 0), \
            "reduce_ds must be -1 or a positive integer"

        if not isinstance(filename, list):
            if ('[' in filename and ']' in filename) or ('glob.glob' in filename):
                filename = eval(filename)
            else:
                filename = [filename]

        self.branches_to_read = [
            'ch_proxy_kin', 'neut_proxy_kin', 'proxy_is_charged',
            'e_t', 'inc_times_node_feat', 'node_feat_sum', 'proxy_em_frac', 'pred_ind',
            'truth_pt', 'truth_eta', 'truth_phi', 'truth_ke', 'truth_class', 'truth_is_charged', 'truth_ind',
            'truth_pt_raw', 'truth_eta_raw', 'truth_ke_raw', 'proxy_pt_raw', 'proxy_ke_raw', 'proxy_eta_raw', 'proxy_phi'
        ]

        self.data_dict = {}; self.n_events = 0
        for fn_i, fn in enumerate(filename):
            f = uproot.open(fn)
            tree = f['event_tree']

            n_events_fni = tree.num_entries
            if (reduce_ds != -1):
                n_events_fni = min(n_events_fni, reduce_ds)
                reduce_ds -= n_events_fni

            self.n_events += n_events_fni
            entry_stop = n_events_fni

            # read the data
            for var in tqdm(self.branches_to_read, desc=f'reading file {fn_i+1}/{len(filename)}'):
                if fn_i == 0:
                    self.data_dict[var] = []

                branch_array = tree[var].array(library='ak', entry_start=0, entry_stop=entry_stop)
                self.data_dict[var].append(branch_array)

            if reduce_ds == 0:
                break

        # stack the data (merging)
        for v_name in self.data_dict.keys():
            self.data_dict[v_name] = ak.concatenate(self.data_dict[v_name], axis=0)

        # reshape vars
        self.vars_to_reshape = ['ch_proxy_kin', 'neut_proxy_kin', 'e_t', 'inc_times_node_feat']
        self.reshape_vars(self.vars_to_reshape)

        # needs to be done after reshaping
        self.apply_pred_indicator_mask()

        # flatten the events
        for k, v in self.data_dict.items():
            if k == 'node_feat_sum':
                self.data_dict[k] = ak.to_numpy(v)
                continue
            v_flat = ak.to_numpy(ak.flatten(v, axis=1))
            self.data_dict[k] = v_flat
        self.n_events = len(self.data_dict['node_feat_sum'])

        gc.collect()
        print(f'\ndataset loaded. Number of examples: {self.n_events}\n')


    def reshape_vars(self, vars_to_reshape):
        _n_particles = ak.to_numpy(ak.count(self.data_dict['truth_pt'], axis=1))
        for k in tqdm(vars_to_reshape, desc='reshaping vars', total=len(vars_to_reshape)):
            v = self.data_dict[k]
            _dim = len(v[0]) // _n_particles[0]
            self.data_dict[k] = ak.unflatten(v, _dim, axis=1)


    def apply_pred_indicator_mask(self):
        ind_mask = self.data_dict['pred_ind'] > self.ind_threshold
        for k, v in self.data_dict.items():
            if k == 'node_feat_sum':
                continue
            self.data_dict[k] = v[ind_mask]


    def __getitem__(self, idx):
        return_dict = {}
        for k, v in self.data_dict.items():
            return_dict[k] = torch.tensor(v[idx])
        return return_dict


    def __len__(self):
        return self.n_events