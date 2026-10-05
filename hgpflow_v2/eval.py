import sys
paths = sys.path
for p in paths:
     if '.local' in p:
             paths.remove(p)


# to ensure reproducibility
import random
import numpy as np
import torch
import pytorch_lightning as pl


import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--inference_input_path', '-i', type=str, required=True)
args = argparse.parse_args()

import yaml, os
with open(args.inference_input_path, 'r') as fp:
    inference_config = yaml.safe_load(fp)


def expand_env_vars(obj):
    # expand ${VAR} in string values; fail loudly if a variable is not set
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    if isinstance(obj, str) and '$' in obj:
        expanded = os.path.expandvars(obj)
        if '$' in expanded:
            raise ValueError(f'unset environment variable in config value: {obj}')
        return expanded
    return obj

inference_config = expand_env_vars(inference_config)

# CUDA; needs to be done before anything CUDA related
# gpu: null keeps the CUDA_VISIBLE_DEVICES set by the caller
if inference_config['init'].get('gpu', None) is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(inference_config['init']['gpu'])


import torch
from .evaluation.inference_helper import InferenceHelper


torch.set_float32_matmul_precision(inference_config['init']['precision'])


inference_helper = InferenceHelper(inference_config['init'])

for inf_dict in inference_config['items']:

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    from numpy.random import default_rng
    RNG = default_rng(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    pl.seed_everything(SEED)

    if inf_dict['run_pred']:
        print('\nrunning prediction on', inf_dict['seg_path'])
        inference_helper.run_prediction(inf_dict)

    rng_state = torch.get_rng_state()
    rng_state_path = inf_dict.get('rng_state_path', None)
    if rng_state_path is None:
        rng_state_path = inf_dict['seg_path'].split('/')[-1].split('.root')[0] + '_rng_state.pt'
    torch.save(rng_state, rng_state_path)

    if inf_dict['run_plot']:
        raise NotImplementedError('plotting not implemented yet')