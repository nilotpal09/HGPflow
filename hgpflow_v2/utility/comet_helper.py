import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CometLogger
from typing import TYPE_CHECKING, Any, Optional, Union
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
import os
if TYPE_CHECKING:
    from comet_ml import ExistingExperiment, Experiment, OfflineExperiment
import warnings


class CometLoggerCustom(CometLogger):
    '''
    I just wanna manually set the experiment keys. comet allows it, but lightning doesn't
    '''
    def __init__(self, api_key: Optional[str] = None, save_dir: Optional[str] = None,
        project_name: Optional[str] = None, rest_api_key: Optional[str] = None,
        experiment_name: Optional[str] = None, experiment_key: Optional[str] = None,
        offline: bool = False, prefix: str = "",
        experiment_key_custom: Optional[str] = None, # custom addition
        **kwargs: Any,
    ):
        super().__init__(api_key, save_dir, project_name, rest_api_key, experiment_name, offline, prefix, **kwargs)
        self._experiment_key_custom = experiment_key_custom

    
    @property
    @rank_zero_experiment
    def experiment(self) -> Union["Experiment", "ExistingExperiment", "OfflineExperiment"]:
        r"""Actual Comet object. To use Comet features in your :class:`~pytorch_lightning.core.LightningModule` do the
        following.

        Example::

            self.logger.experiment.some_comet_function()

        """
        if self._experiment is not None and self._experiment.alive:
            return self._experiment

        if self._future_experiment_key is not None:
            os.environ["COMET_EXPERIMENT_KEY"] = self._future_experiment_key

        from comet_ml import ExistingExperiment, Experiment, OfflineExperiment

        try:
            if self.mode == "online":
                if self._experiment_key is None:
                    self._experiment = Experiment(api_key=self.api_key, project_name=self._project_name, **self._kwargs)
                    self._experiment_key = self._experiment.get_key()
                else:
                    # custom addition
                    if self._experiment_key_custom is not None:
                        self._experiment = Experiment(
                            api_key=self.api_key,
                            project_name=self._project_name,
                            experiment_key=self._experiment_key_custom,
                            **self._kwargs,
                        )
                    else:
                        self._experiment = ExistingExperiment(
                            api_key=self.api_key,
                            project_name=self._project_name,
                            previous_experiment=self._experiment_key,
                            **self._kwargs,
                        )
            else:
                self._experiment = OfflineExperiment(
                    offline_directory=self.save_dir, project_name=self._project_name, **self._kwargs
                )
            self._experiment.log_other("Created from", "pytorch-lightning")
        finally:
            if self._future_experiment_key is not None:
                os.environ.pop("COMET_EXPERIMENT_KEY")
                self._future_experiment_key = None

        if self._experiment_name:
            self._experiment.set_name(self._experiment_name)

        return self._experiment




def save_plot(fig, name, comet_logger=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    if comet_logger is not None:
        fig.canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(h), int(w), 3)
        comet_logger.experiment.log_image(
            image_data=image,
            name=name,
            overwrite=False, 
            image_format="png",
        )
    else:
        fig.savefig(f'plot_dump/{name}.png', bbox_inches='tight')
    plt.close(fig)


def log_parameters(comet_logger, config, prefix=''):
    for k, v in config.items():
        if isinstance(v, dict):
            log_parameters(comet_logger, v, prefix=f'{prefix}{k}.')
        else:
            comet_logger.experiment.log_parameter(f'{prefix}{k}', v)