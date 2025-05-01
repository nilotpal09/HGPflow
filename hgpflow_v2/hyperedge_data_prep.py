import sys
paths = sys.path
for p in paths:
     if '.local' in p:
             paths.remove(p)

import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--inference_input_path', '-i', type=str, required=True)
args = argparse.parse_args()

import yaml, os, glob
with open(args.inference_input_path, 'r') as fp:
    inference_config = yaml.safe_load(fp)

# CUDA; needs to be done before anything CUDA related
os.environ['CUDA_VISIBLE_DEVICES'] = str(inference_config['init']['gpu'])


import torch
from .evaluation.hyperedge_data_prep_helper import HyperedgeDataPrepHelper


torch.set_float32_matmul_precision(inference_config['init']['precision'])


inference_helper = HyperedgeDataPrepHelper(
    init_config=inference_config['init'])

for inf_dict in inference_config['items']:

    seg_paths = inf_dict['seg_path']
    if not isinstance(seg_paths, list):
        if ('[' in seg_paths and ']' in seg_paths) or ('glob.glob' in seg_paths):
            seg_paths = eval(seg_paths)
        else:
            seg_paths = [seg_paths]

    output_filepath_base = inf_dict.get('output_filepath')

    for seg_path in seg_paths:
        inf_dict['seg_path'] = seg_path
        inf_dict['output_filepath'] = output_filepath_base
        print('\nrunning stage1 inference on', inf_dict['seg_path'])
        inference_helper.run_inference(inf_dict)