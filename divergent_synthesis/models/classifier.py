import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, sys, pdb, re

from divergent_synthesis.losses.loss import LossContainer
sys.path.append('../')
from omegaconf import OmegaConf, ListConfig
from divergent_synthesis.models.model import Model
from divergent_synthesis.modules import encoders
from torch import distributions as dist
from divergent_synthesis import losses 


class Classifier(Model):
    def __init__(self, config: OmegaConf=None, **kwargs):
        """
        Base class for single domain classifiers. Configuration file must include:
        - classifier 
        - training

        Args:
            config (Config): full configuration
            **kwargs:
        """
        # manage config
        config = config or OmegaConf.create()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        config_checkpoint = config.get('config_checkpoint')
        super().__init__(config=config)

        # input config
        self.input_shape = config.get('input_shape') or kwargs.get('input_shape')

        # create classifier
        assert config.get('classifiers') is not None
        classifiers = {}
        for task_name, classif_config in config.classifiers.items():
            classif_config.args = classif_config.get('args', {})
            classif_config.args.input_shape = self.input_shape
            classif_config.args.target_shape = classif_config.dim
            classifiers[task_name] = getattr(encoders, classif_config.get('type'))(classif_config.get('args', {}))
        self.classifiers = nn.ModuleDict(classifiers)

        # loss
        config.training = config.get('training')
        loss_config = config.training.get('classification', OmegaConf.create())
        if isinstance(loss_config, ListConfig):
            loss = LossContainer()
            for config_tmp in loss_config:
                loss_tmp = getattr(losses, config_tmp['type'])(**config_tmp.get('args', {}))
                loss.append(loss_tmp, config_tmp.get('weight', 1.0))
        else:
            loss = getattr(losses, loss_config.get('type', 'LogDensity'))(**loss_config.get('args', {}))
        self.classif_loss = loss

        # load from checkpoint
        if config_checkpoint:
            self.import_checkpoint(config_checkpoint)

        # record configs
        self.save_config(self.config)

    def configure_optimizers(self):
        optimizer_config = self.config.training.get('optimizer', {'type':'Adam'})
        optimizer_args = optimizer_config.get('args', {'lr':1e-4})
        parameters = self.get_parameters(self.config.training.get('optim_params'))
        optimizer = getattr(torch.optim, optimizer_config['type'])(parameters, **optimizer_args)
        if self.config.training.get('scheduler'):
            scheduler_config = self.config.training.get('scheduler')
            scheduler_args = scheduler_config.get('args', {})
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_config.type)(optimizer, **scheduler_args)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss/valid"}
        else:
            return optimizer

    # External methods
    def classify(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # in lightning, forward defines the prediction/inference actions
        outs = {}
        for task, classifier in self.classifiers.items():
            outs[task] = classifier(x)
        return outs

    def trace(self, x: torch.Tensor, sample: bool = False):
        trace_model = {'classifier_'+t: {} for t in self.classifiers.keys()}
        if isinstance(x, (tuple, list)):
            x, y = x
        for k, t in self.classifiers.items():
            out = t(x, trace = trace_model['classifier_'+k])
        trace = {}
        trace['embeddings'] = {**trace.get('embeddings', {})}
        trace['histograms'] = {**trace.get('histograms', {})}
        return trace

    def loss(self, batch, x, y, drop_detail = False, **kwargs):
        # reconstruction losses
        return self.classif_loss(x, y, drop_detail=drop_detail)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y = batch
        loss = 0.
        losses = {}
        for task, classifier in self.classifiers.items():
            classif_out = classifier(x)
            loss_tmp, losses_tmp = self.loss(batch, classif_out, y[task], epoch=self.trainer.current_epoch, drop_detail=True)
            loss = loss + loss_tmp
            losses = {**losses, **{k+"_"+task: v for k, v in losses_tmp.items()}}
        self.log_losses(losses, "train", prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        batch, y = batch
        loss = 0.
        losses = {}
        for task, classifier in self.classifiers.items():
            classif_out = classifier(batch)
            loss_tmp, losses_tmp = self.loss(batch, classif_out, y[task], epoch=self.trainer.current_epoch, drop_detail=True)
            loss = loss + loss_tmp
            losses = {**losses, **{k+"_"+task: v for k, v in losses_tmp.items()}}
        self.log_losses(losses, "valid", prog_bar=True)
        return loss