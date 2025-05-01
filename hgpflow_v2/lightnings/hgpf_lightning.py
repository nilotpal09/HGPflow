import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm

import numpy as np
import matplotlib as mpl
mpl.use('Agg')

from ..utility import metrics, live_plotting
from ..utility.custom_lr_scheduler import CustomLRScheduler
from ..utility.comet_helper import save_plot
from ..utility.helper_dicts import class_mass_dict

from ..models.hgpflow_model import HGPFlowModel
from .hgpf_lightning_supattn import HGPFLightningSA
from .hgpf_lightning_iterativerefiner import HGPFLightningIR
from .hgpf_lightning_hyperedge import HGPFLightningHyperedge
from .lightning_helper import get_val_step_perf

import ray


class HGPFLightning(LightningModule):

    def __init__(self, config_v, config_ms1, config_ms2=None, config_t=None, comet_logger=None):
        super().__init__()
        
        self.config_v = config_v
        self.config_ms1 = config_ms1
        self.config_ms2 = config_ms2
        self.config_t = config_t

        self.net = HGPFlowModel(
            config_v, config_ms1, config_ms2, class_mass_dict)

        self.comet_logger = comet_logger

        if self.config_t is not None: # CHANGE IT LATER
            # ray
            self.n_ray = self.config_t.get('n_ray', 0)
            if self.n_ray > 0:
                ray.init(num_cpus=self.n_ray, include_dashboard=False)
                print(f'ray initialized with {self.n_ray} cpus')

            lightning_dict = {
                'sup_attn': HGPFLightningSA,
                'iterative_refiner': HGPFLightningIR,
                'hyperedge_model': HGPFLightningHyperedge
            }

            self.model_type = 'hyperedge_model'
            if 'hg_model' in config_t['train_components']:
                self.model_type = config_ms1['hg_model']['type']
                self.metrics = metrics.Metrics(config_t, self.n_ray)

            self._lightning = lightning_dict[self.model_type](
                self, config_v, config_ms1, config_ms2, config_t)

            # validation step logger
            self.validation_step_perf = get_val_step_perf(
                self.model_type, config_v, config_ms1, config_ms2, config_t)
            self.norms2store = []

        self.save_hyperparameters()


    def train_dataloader(self):
        return self._lightning.train_dataloader()

    def val_dataloader(self):
        return self._lightning.val_dataloader()


    def training_step(self, batch, batch_idx):
        return self._lightning.training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self._lightning.validation_step(batch, batch_idx)


    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.net, norm_type=2)
        self.norms2store.append(norms['grad_2.0_norm_total'])


    def set_comet_logger(self, comet_logger):
        self.comet_logger = comet_logger


    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=self.config_t['learning_rate'])

        if self.config_t.get('lr_scheduler', None) == None:
            return optimizer

        elif self.config_t['lr_scheduler']['name'] == 'CustomLRScheduler':
            warm_start_epochs = self.config_t['lr_scheduler']['warm_start_epochs']
            cosine_epochs = self.config_t['lr_scheduler']['cosine_epochs']
            eta_min = self.config_t['lr_scheduler']['eta_min']
            last_epoch = self.config_t['lr_scheduler']['last_epoch']
            max_epoch = self.config_t['num_epochs']
            if (cosine_epochs > 0 and cosine_epochs < 1) or (warm_start_epochs > 0 and warm_start_epochs < 1):
                max_epoch = self.config_t['num_epochs']
            scheduler = CustomLRScheduler(optimizer, warm_start_epochs, cosine_epochs, eta_min, last_epoch, max_epoch)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        else:
            raise NotImplementedError(f"lr_scheduler {self.config_t['lr_scheduler']['name']} not implemented")


    def on_train_epoch_end(self):
        if self.config_t.get('lr_scheduler', None) is not None:
            self.log('lr', self.lr_schedulers().get_last_lr()[0])
            if not self.automatic_optimization:
                self.lr_schedulers().step()


    def on_validation_epoch_end(self):
        logs = {}
        for key in self.validation_step_perf.log_dicts[0].keys():
            avg_loss = np.hstack([x[key] for x in self.validation_step_perf.log_dicts]).mean()
            logs[key] = avg_loss
        self.log_dict(logs)

        if hasattr(self.validation_step_perf, 'hg_summaries'):
            fig, log_dict = live_plotting.plot_hg_summary(self.validation_step_perf.hg_summaries)
            for k, v in log_dict.items():
                self.log(f"val_{k}", v)
            save_plot(fig, 'inc_performance', self.comet_logger)

        if hasattr(self.validation_step_perf, 'dep_es'):
            fig, log_dict = live_plotting.plot_inc_dep_e(self.validation_step_perf.dep_es)
            for k, v in log_dict.items():
                self.log(f"val_{k}", v)
            save_plot(fig, 'inc_dep_e', self.comet_logger)

        if hasattr(self.validation_step_perf, 'eds'):
            for i, plot_dict in enumerate(self.validation_step_perf.eds):
                fig = live_plotting.plot_inc_ed(plot_dict)
                save_plot(fig, f'inc_event_display_{plot_dict["idx"]}', self.comet_logger)

        if hasattr(self.validation_step_perf, 'hyperedge_dicts'):
            fig_class, fig_regression = live_plotting.plot_hyperedge(
                self.validation_step_perf.hyperedge_dicts, 
                apply_truth_ind_mask=self.config_t.get('apply_truth_ind_mask_on_live_plots', False))
            save_plot(fig_class, 'hyperedge_class', self.comet_logger)
            if fig_regression is not None:
                save_plot(fig_regression, 'hyperedge_regression', self.comet_logger)

        self.validation_step_perf.reset()