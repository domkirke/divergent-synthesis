import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, sys, pdb, re
from copy import deepcopy
from torch import distributions as dist
sys.path.append('../')
from omegaconf import OmegaConf, ListConfig
from divergent_synthesis.models.model import Model, ConfigType
from divergent_synthesis import models, losses
import inspect

def count_positional_args_required(func):
    signature = inspect.signature(func)
    empty = inspect.Parameter.empty
    total = 0
    for param in signature.parameters.values():
        if param.default is empty:
            total += 1
    return total


def get_loss(config):
    if isinstance(config, ListConfig):
        loss =[get_loss(l) for l in config]
        return sum(loss[1:], loss[0])
    else:
        loss_type = config.get('type')
        if loss_type is None:
            raise ValueError("key missing in loss configuration : type")
        loss_args = config.get('args', {})
        loss = getattr(losses, loss_type)(**loss_args)
        if config.get('weight'):
            loss = config.weight * loss
        return loss


import os
def get_model_from_path(model_path: str):
    if os.path.isfile(model_path):
        assert os.path.splitext(model_path)[1] == ".ckpt", "%s does not seem to be a valid file"%model_path
        yaml_paths = [os.path.dirname(model_path), os.path.abspath(os.path.dirname(model_path)+"/..")]
    else:
        model_path = os.path.join(model_path, "last.ckpt")
        yaml_paths = [os.path.dirname(model_path)]
        assert os.path.isfile(model_path), "model not found in folder %s"%model_path
    yaml_files = []
    for yp in yaml_paths:
        yaml_files.extend([os.path.join(yp, f) for f in os.listdir(yp)])
    yaml_files = list(filter(lambda x: os.path.splitext(x)[1] == ".yaml", yaml_files))
    if len(yaml_files) > 1:
        print('[Warning] ambiguous ; retrieved yaml file %s'%yaml_files[0])
    config = OmegaConf.load(os.path.abspath(yaml_files[0]))
    model = getattr(models, config.model.type)(config=config.model)
    model = model.load_from_checkpoint(model_path, config=config.model)
    return model, config


class DivergentGenerative(Model):
    def __init__(self, config=ConfigType, **kwargs):
        config = config or OmegaConf.create()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        config_checkpoint = config.get('config_checkpoint')
        super().__init__(config=config)

        # init generator
        generator_config = config['generator']
        self.original_config = None
        if generator_config.get('path') is not None:
            self.generator, self.original_config = get_model_from_path(generator_config.path)
        else:
            self.generator = getattr(models, config.model.type)(config=config.model)

        # init losses
        self.losses = get_loss(config.training.losses)
        self.current_loss_idx = None
        if config.training.get('balance'):
            self.current_loss_idx = 0

        self.automatic_optimization = False
        self.generator_history = None

    def configure_optimizers(self):
        optimizer_config = self.config.training.get('optimizer', {'type':'Adam'})
        optimizer_args = optimizer_config.get('args', {'lr':1e-4})
        parameters = self.get_parameters(self.config.training.optimizer.get('params'))
        optimizer = getattr(torch.optim, optimizer_config['type'])(parameters, **optimizer_args)
        if self.config.training.get('scheduler'):
            scheduler_config = self.config.training.get('scheduler')
            scheduler_args = scheduler_config.get('args', {})
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_config.type)(optimizer, **scheduler_args)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss/valid"}
        else:
            return optimizer

    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: int = 0) -> None:
        if self.current_loss_idx is not None:
            self.current_loss_idx = (self.current_loss_idx + 1) % len(self.losses)
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def get_loss(self, generations, original, idx, y, drop_detail=False, force_all=False):
        if self.config.training.get('balance') and (not force_all):
            current_loss, current_weight = self.losses[self.current_loss_idx]
        else:
            current_loss, current_weight = self.losses, 1.0
        if drop_detail:
            current_loss, current_losses = current_loss(generations, original, y, drop_detail=drop_detail)
            current_loss = current_loss * current_weight
            return current_loss, current_losses
        else:
            return current_weight * current_loss(generations, original, y, drop_detail=drop_detail)

    def training_step(self, batch, batch_idx):
        self.optimizers().zero_grad()
        self.losses.zero_grad()
        # retain parameters in history
        self.generator_history = deepcopy(self.generator.state_dict())
        # with torch.autograd.detect_anomaly():
        if isinstance(batch, (tuple, list)):
            batch, y = batch
        generations = self.generator.sample_from_prior(batch.shape[0])[:, 0]
        loss, losses = self.get_loss(generations, batch, batch_idx, y, drop_detail=True)
        self.log_losses(losses, "train", prog_bar=True)
        if self.current_loss_idx is not None:
            self.log_losses(self.current_loss_idx, "current_loss", prog_bar=True)
        self.manual_backward(loss)
        with open('log.txt', 'a') as f:
            # f.write('\nEpoch %s\n'%batch_idx)
            # f.write('GRADIENTS\n')
            # for k, v in self.named_parameters():
            #     if v.grad is not None:
            #         f.write(f"{k} \t {v.min()}; {v.max()}; {v.mean()}; {v.std()}\n")        
            #     else:
            #         f.write(f'{k}\tNone\n')
            self.optimizers().step()
            # f.write('PARAMETERS AFTER STEP\n')
            # for k, v in self.named_parameters():
            #     f.write(f"{k} \t {v.min()}; {v.max()}; {v.mean()}; {v.std()}\n")        
        if True in [torch.isnan(v).any() for v in self.generator.parameters()]:
            print('nan after update')
            self.generator.load_state_dict(self.generator_history)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)):
            batch, y = batch
        generations = self.generator.sample_from_prior(batch.shape[0], temperature=[1.0])[:, 0]
        loss, losses = self.get_loss(generations, batch, batch_idx, y, drop_detail=True, force_all=True)
        self.log_losses(losses, "val", prog_bar=True)
        if self.current_loss_idx is not None:
            self.log_losses(self.current_loss_idx, "current_loss", prog_bar=True)
        return loss

    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        return self.generator.sample_from_prior(n_samples, temperature, sample)

        
