import comet_ml

# remove local paths, so that we don't use any
import sys, os
paths = sys.path
for p in paths:
    if '.local' in p:
            paths.remove(p)

import argparse
from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument('--config_path_var', '-cv', type=str, required=False)
argparser.add_argument('--config_path_model_stage1', '-cms1', type=str, required=False)
argparser.add_argument('--config_path_model_stage2', '-cms2', type=str, required=False)
argparser.add_argument('--config_path_train', '-ct', type=str, required=True)
argparser.add_argument('--exp_key', '-ekey', type=str, required=False)
argparser.add_argument('--debug_mode', '-d', action='store_true')
argparser.add_argument('--precision', '-p', type=str, required=False, default='medium')
argparser.add_argument('--gpu', '-g', type=str, required=False, default='0')

args = argparser.parse_args()
config_path_v = args.config_path_var
config_path_ms1 = args.config_path_model_stage1
config_path_ms2 = args.config_path_model_stage2
config_path_t = args.config_path_train
debug_mode = args.debug_mode
exp_key = args.exp_key
precision = args.precision

assert config_path_t is not None
stage1_condition = (config_path_v is not None) and (config_path_ms1 is not None) and (config_path_ms2 is None)
stage2_condition = (config_path_v is None) and (config_path_ms1 is None) and (config_path_ms2 is not None)
assert stage1_condition or stage2_condition, \
    "stage1 and stage2 are mutually exclusive\n" + \
    "stage1: need exactly --config_path_var (-cv), --config_path_model_stage1 (-cms1)\n" + \
    "stage2: need exactly --config_path_model_stage2 (-cms2)"
    

# need to set CUDA_VISIBLE_DEVICES before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.system('nvidia-smi')

import yaml, glob, random, string, shutil
import torch

from .lightnings.hgpf_lightning import HGPFLightning
from .utility.comet_helper import CometLoggerCustom
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import Trainer

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# set precision
torch.set_float32_matmul_precision(precision)

# configs
with open(config_path_t, 'r') as fp:
    config_t = yaml.safe_load(fp)

if stage1_condition:
    print("\033[96mTraining stage 1\033[00m")
    with open(config_path_v, 'r') as fp:
        config_v = yaml.safe_load(fp)
    with open(config_path_ms1, 'r') as fp:
        config_ms1 = yaml.safe_load(fp)
    config_ms2 = None
else:
    print("\033[96mTraining stage 2 with frozen stage 1\033[00m")
    with open(config_path_ms2, 'r') as fp:
        config_ms2 = yaml.safe_load(fp)

    # get config_v and config_ms1 from frozen stage1
    config_path_v = config_t['config_path_v']
    with open(config_path_v, 'r') as fp:
        config_v = yaml.safe_load(fp)
    config_path_ms1 = config_t['config_path_ms1']
    with open(config_path_ms1, 'r') as fp:
        config_ms1 = yaml.safe_load(fp)

# model
lightning_model = HGPFLightning(config_v, config_ms1, config_ms2, config_t)

# for saving checkpoints for best 3 models (according to val loss) and last epoch
checkpoint_callback = ModelCheckpoint(
    monitor='val_total_loss',
    mode='min',
    every_n_train_steps=0,
    every_n_epochs=1,
    train_time_interval=None,
    save_top_k=3,
    save_last= True,
    filename='{epoch}-{val_total_loss:.4f}')


if debug_mode:
    trainer = Trainer(
        max_epochs = config_t['num_epochs'],
        accelerator = config_t['device'],
        devices = config_t['num_devices'],
        default_root_dir = config_t["base_root_dir"],
        callbacks = [checkpoint_callback],
        check_val_every_n_epoch = config_t['eval_every_n_epoch'],
        gradient_clip_val=1.0 if lightning_model.automatic_optimization else None,
        # precision="16-mixed",
    )

else:
    if exp_key is None:
        exp_key  = f'{config_t["run_name"]}xxx'
        exp_key += ''.join(random.choices(string.ascii_lowercase + string.digits, k=32-len(exp_key)))

        dst = f'{config_t["base_root_dir"]}/{config_t["project_name"]}/{exp_key}'
        Path(dst).mkdir(parents=True, exist_ok=True)

        new_config_path_t = os.path.join(dst, 'config_t.yml')
        shutil.copyfile(config_path_t, new_config_path_t)

        new_config_path_v = os.path.join(dst, 'config_v.yml')
        shutil.copyfile(config_path_v, new_config_path_v)

        new_config_path_ms1 = os.path.join(dst, 'config_ms1.yml')
        shutil.copyfile(config_path_ms1, new_config_path_ms1)

        if config_path_ms2 is not None:
            new_config_path_ms2 = os.path.join(dst, 'config_ms2.yml')
            shutil.copyfile(config_path_ms2, new_config_path_ms2)

    comet_logger = CometLoggerCustom(
        api_key = os.environ["COMET_API_KEY"],
        project_name = config_t["project_name"], # hgpflow_v2
        workspace = os.environ["COMET_WORKSPACE"], # user_name
        experiment_name = config_t["run_name"],
        experiment_key_custom = exp_key
    )

    lightning_model.set_comet_logger(comet_logger)
    comet_logger.experiment.log_asset(config_path_v, file_name='config_v')
    comet_logger.experiment.log_asset(config_path_t, file_name='config_t')
    comet_logger.experiment.log_asset(config_path_ms1, file_name='config_ms1')
    if config_path_ms2 is not None:
        comet_logger.experiment.log_asset(config_path_ms2, file_name='config_ms2')

    comet_logger.experiment.log_parameter('ekey', exp_key)
    comet_logger.experiment.log_parameter(
        'experiment_path', f'{config_t["base_root_dir"]}/{config_t["project_name"]}/{exp_key}')

    # log files
    dirs2log = ['.', 'models', 'utility']
    for d in dirs2log:
        all_files = glob.glob(f'{d}/*.py')
        for fpath in all_files:
            comet_logger.experiment.log_asset(fpath, file_name=f'{d}/{fpath}')

    trainer = Trainer(
        max_epochs = config_t['num_epochs'],
        accelerator = config_t['device'],
        devices = config_t['num_devices'],
        default_root_dir = config_t["base_root_dir"],
        callbacks = [checkpoint_callback, TQDMProgressBar(refresh_rate=100)],
        check_val_every_n_epoch = config_t['eval_every_n_epoch'],
        log_every_n_steps = 1,
        logger = comet_logger,
        gradient_clip_val=1.0 if lightning_model.automatic_optimization else None,
    )

# run trainer
trainer.fit(lightning_model, ckpt_path=config_t['resume_from_checkpoint'])
