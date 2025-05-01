import numpy as np
from torch.utils.data import DataLoader

from .dataset_mini import PflowDatasetMini, collate_fn_mini, PflowSamplerMini
from .dataset_hyperedge import PflowDatasetHyperedge



ds_dict = {
    'mini': PflowDatasetMini,
    'hyperedge': PflowDatasetHyperedge
}

collate_fn_dict = {
    'mini': collate_fn_mini,
}

sampler_dict = {
    'mini': PflowSamplerMini
}


def get_dataloader(dataset_type, ds_kwargs, sampler_kwargs, loader_kwargs):
    dataset = ds_dict[dataset_type](**ds_kwargs)

    collate_fn = collate_fn_dict.get(dataset_type, None)
    
    batch_sampler = None
    if dataset_type in sampler_dict:
        remove_idxs = []
        if sampler_kwargs['remove_idxs'] == True:
            remove_idxs = np.where(dataset.n_particles > sampler_kwargs['config_v']['max_particles'])[0]
            remove_idxs = np.unique(np.concatenate([remove_idxs, np.where(dataset.n_nodes < 2)[0]]))

            # CLIP BAD IDXS HACK
            if sampler_kwargs['config_v'].get('filter_clip_bad_idxs', False):
                track_pt_lens = np.array([len(x) for x in dataset.data_dict['track_pt']])
                track_particle_idxs = np.array([len(x) for x in dataset.data_dict['track_particle_idx']])
                bad_idxs = np.where(track_pt_lens != track_particle_idxs)[0]
                print(f'\033[96m[INFO] {len(bad_idxs)} bad idxs found in dataset \033[0m')
                remove_idxs = np.unique(np.concatenate([remove_idxs, bad_idxs]))

        if sampler_kwargs['apply_cells_threshold']:
            n_cells_array = dataset.n_cells
            batch_sampler = sampler_dict[dataset_type](
                dataset.n_nodes, batch_size=sampler_kwargs['batch_size'], remove_idxs=remove_idxs,
                n_cells_array=n_cells_array, n_cells_threshold=sampler_kwargs['n_cells_threshold'])
        else:
            batch_sampler = sampler_dict[dataset_type](
                dataset.n_nodes, batch_size=sampler_kwargs['batch_size'], remove_idxs=remove_idxs)


    loader_kwargs['collate_fn'] = collate_fn
    loader_kwargs['batch_sampler'] = batch_sampler

    loader = DataLoader(dataset, **loader_kwargs)

    return loader
