import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, sys, pdb, re
from torch import distributions as dist
sys.path.append('../')
from omegaconf import OmegaConf, ListConfig
from divergent_synthesis.modules import encoders
from .model import Model
from .gans import GAN
from divergent_synthesis import models, losses
from .. import discriminators
from divergent_synthesis.losses import get_regularization_loss, priors
from divergent_synthesis.utils import checklist, checksize


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
    model = model.load_from_checkpoint(model_path)
    return model, config


class DivergentExplorer(GAN):
    def __init__(self, config=OmegaConf, **kwargs):
        config = config or OmegaConf.create()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        config_checkpoint = config.get('config_checkpoint')
        Model.__init__(self, config=config)

        # init generator
        generator_config = config['generator']
        self.generator, original_config = get_model_from_path(generator_config.path)
        self.config.latent = original_config.model.latent
        assert hasattr(self.generator, "encode"), "Explorer needs a generator with the encode function"

        # init latent generator
        explorer_config = config['explorer']
        explorer_config.args.input_shape = self.config.latent.dim
        explorer_config.args.target_shape = self.config.latent.dim
        self.explorer = getattr(encoders, explorer_config.type)(explorer_config.args)

        # init latent discriminator
        disc_config = OmegaConf.create(self.config)
        disc_config['input_shape'] = self.config.latent.dim
        discriminator_class = config.discriminator.type
        self.discriminator = getattr(discriminators, discriminator_class)(disc_config)

        # init losses
        self.current_loss_idx = None
        if config.training.get('balance'):
            self.current_loss_idx = 0

        reg_config = config.training.get('regularization', OmegaConf.create())
        self.regularization_loss = get_regularization_loss(reg_config)
        self.prior = getattr(priors, config.training.get('prior', "isotropic_gaussian"))

        self.automatic_optimization = False

    def configure_optimizers(self):
        lr = checklist(self.config.training.get('lr', 1e-4), n=2)
        if self.config.training.mode == "wasserstein":
            betas = (0.5, 0.999)
        else:
            betas = (0.9, 0.999)
        gen_optim = torch.optim.Adam(self.explorer.parameters(), lr=lr[0] or 1e-4, betas=betas)
        disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr[1] or 1e-4, betas=betas)
        self.config.training.balance = self.config.training.get('balance')
        return gen_optim, disc_optim

    def on_train_epoch_start(self):
        if self.config.training.balance is not None:
            self._loss_phase = 0
            self._loss_counter = 0
        else:
            self._loss_phase = None
            self._loss_counter = None
    
    def on_validation_epoch_start(self):
        self._loss_phase = None

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        if self.config.training.balance is not None:
            self._loss_counter += 1
            if self._loss_counter >= self.config.training.balance[self._loss_phase]:
                self._loss_phase = (self._loss_phase + 1) % 2
                self._loss_counter = 0

    def get_beta(self):
        beta = self.config.training.beta if self.config.training.beta is not None else 1.0
        if self.config.training.warmup:
            beta_schedule_type = self.config.training.get('beta_schedule_type', "epoch")
            if beta_schedule_type == "epoch":
                beta = min(int(self.trainer.current_epoch) / self.config.training.warmup, beta)
            elif beta_schedule_type == "batch":
                beta = min(int(self.trainer.global_step) / self.config.training.warmup, beta)
        return beta

    def training_step(self, batch, batch_idx):
        x, y = batch
        # initialize optimizers
        g_opt, d_opt = self.optimizers()
        g_opt.zero_grad()
        d_opt.zero_grad()
        # generate
        prior = self.sample_prior(batch=x, y=y)
        z = prior['z']
        y = prior['y']
        z_real = self.generator.encode(x)
        z_fake = self.explorer(z)
        g_losses = d_losses = None
        loss = 0
        if self._loss_phase == 0 or self._loss_phase is None:
            # update discriminator
            d_loss, d_losses = self.optimize_discriminator(z_fake, x=z_real, y=y, batch_idx=batch_idx)
            loss = loss + d_loss
        if self._loss_phase == 1 or self._loss_phase is None:
            # update generator
            g_loss, g_losses = self.optimize_generator(z_fake, x=z_real, y=y, batch_idx=batch_idx)
            loss = loss - g_loss
        # regularization loss
        reg_loss = self.regularization_loss(z_fake, self.generator.prior(z_fake.shape, device=z_fake.device))
        loss = loss + self.get_beta() * reg_loss
        self.log_losses({'reg_loss': loss.detach().item()}, "train")
        # log losses
        if g_losses is not None:
            self.log_losses(g_losses, "train")
        if d_losses is not None:
            self.log_losses(d_losses, "train")
        self.log("loss/train", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # generate
        prior = self.sample_prior(batch=x, y=y)
        z = prior['z']
        y = prior['y']
        z_real = self.generator.encode(x)
        z_fake = self.explorer(z)
        g_losses = d_losses = None
        d_real, hidden_real = self.discriminate(x=z_real, y=y, z=z, return_hidden=True)
        d_fake, hidden_fake = self.discriminate(x=z_fake, y=y, z=z, return_hidden=True)
        d_loss, d_losses = self.discriminator_loss(self.discriminator, z_real, z_fake, d_real, d_fake, z=z, y=y, drop_detail=True)
        # get generator loss
        g_loss, g_losses = self.generator_loss(self.generator, z_real, z_fake, d_fake, z=z, y=y, hidden=[hidden_real, hidden_fake], drop_detail=True)
        self.log_losses(g_losses, "valid")
        self.log_losses(d_losses, "valid")
        loss = (g_loss or 0) + (d_loss or 0)
        # regularization loss
        reg_loss = self.regularization_loss(z_fake, self.generator.prior(z_fake.shape, device=z_fake.device))
        loss = loss + self.get_beta() * reg_loss
        self.log_losses({'reg_loss': loss.detach().item()}, "train")
        self.log("loss/valid", loss, prog_bar=False)
        return loss

    def sample_prior(self, batch=None, shape=None, y=None):
        """Samples the GAN's prior to obtain latent positions."""
        if batch is not None:
            batch_len = len(batch.shape) - len(self.generator.decoder.target_shape)
            batch_shape = batch.shape[:batch_len]
        else:
            batch_shape = shape
        y = None
        z = self.prior((*batch_shape, *checksize(self.config.latent.dim))).sample()
        return {'z': z.to(self.device), 'y':y}

    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        """Generates data by sampling the GAN's prior and decoding the correspondig data"""
        temperature = checklist(temperature)
        generations = []
        for t in temperature:
            z = torch.randn((n_samples, self.config.latent.dim), device=self.device) * t
            y = {}
            if self.config.get('classes') is not None:
                for task, task_params in self.config.classes.items():
                    y[task] = torch.randint(0, task_params['dim'], (n_samples,), device=self.device)
            z = self.explorer(z)
            x = self.generator.decode(z, y=y)
            if isinstance(x, dist.Distribution):
                if sample:
                    x = x.sample()
                else:
                    x = x.mean
            generations.append(x)
        return torch.stack(generations, 1)

    def generate(self, x=None, y=None, z=None, trace=None):
        return self.generator.decode(z=z, y=y, trace=trace)

    def trace(self, x: torch.Tensor, sample: bool = False):
        if isinstance(x, (tuple, list)):
            x, y = x
        trace_model = {'generator': {}, 'discriminator' : {}, "explorer": {}, "latent": {}}
        z = self.sample_prior(x)
        z = self.explorer(z['z'], trace=trace_model['explorer'])
        if isinstance(z, dist.Normal):
            for z_i in range(z.mean.shape[-1]):
                trace_model['latent']['latent_mean/dim_%d'%z_i] = z[..., z_i]
                trace_model['latent']['latent_std/dim_%d'%z_i] = z[..., z_i]
        else:
            for z_i in range(z.shape[-1]):
                trace_model['latent']['latent/dim_%d'%z_i] = z[..., z_i]
        out = self.generate(x=x, y=y, z=z, trace=trace_model['generator'])
        _ = self.discriminate(x=z, y=y, trace=trace_model['discriminator'])
        trace = {}
        trace['histograms'] = trace_model
        return trace