from matplotlib import path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.distributions as dist, sys, pdb, random
sys.path.append("../")
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from divergent_synthesis import modules
from divergent_synthesis.models import Model
from divergent_synthesis.modules import encoders
from divergent_synthesis.utils import checklist
from divergent_synthesis import losses
from divergent_synthesis.losses import priors


def parse_additional_losses(config):
    if config is None:
        return None
    if isinstance(config, list):
        return [parse_additional_losses[c] for c in config]
    return getattr(losses, config.type)(**config.get('args', {}))


class GAN(Model):
    gan_modes = ['logistic', 'hinge', 'wasserstein', 'softplus', 'softplus-ns']

    def __init__(self, config=None):
        """
        Main class for Generative logisticersarial Networks (GAN). The configuration must include:
        - generator
        - discriminator
        - latent
        - training

        Args:
            config (OmegaConf): configuration file for GANs.
        """
        # manage config
        config = config or OmegaConf.create()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        config_checkpoint = config.get('config_checkpoint')
        super().__init__(config=config)

        input_shape = config.get('input_shape')
        # setup latent
        config.latent = config.get('latent')
        self.prior = getattr(priors, config.latent.get('prior', "isotropic_gaussian"))
        # setup generator
        config.generator = config.get('generator')
        if config.latent.get('dim') and config.generator.args.get('input_shape') is None:
            config.generator.args.input_shape = config.latent.dim
        config.generator.args.target_shape = input_shape
        self.init_generator(config.generator)
        if config.latent.get('dim') is None:
            config.latent.dim = self.generator.input_shape
        # setup discriminator
        config.discriminator = config.get('discriminator')
        config.discriminator.args.input_shape = input_shape
        self.init_discriminator(config.discriminator)
        self.input_augmentations = None
        if config.discriminator.get('input_augmentations'):
            input_augmentations = []
            for m in config.discriminator.input_augmentations:
                input_augmentations.append(getattr(modules, m['type'])(**m.get('args', {})))
            self.input_augmentations = nn.Sequential(*input_augmentations)
        # setup training
        config.training = config.get('training')
        config.training.mode = config.training.get('mode', 'logistic')
        assert config.training.mode in self.gan_modes
        self.automatic_optimization = False
        self.reconstruction_losses = parse_additional_losses(config.training.get('rec_losses'))
        self.training_epoch = config.get('training_epoch', 0)
        # load from checkpoint
        if config_checkpoint:
            self.import_checkpoint(config_checkpoint)
        # record configs
        self.save_config(self.config)

    def init_generator(self, config: OmegaConf) -> None:
        generator_type = config.type or "DeconvEncoder"
        self.generator = getattr(encoders, generator_type)(config.args)

    def init_discriminator(self, config: OmegaConf) -> None:
        disc_type = config.type or "ConvEncoder"
        config.args.target_shape = 1
        self.discriminator = getattr(encoders, disc_type)(config.args)

    def configure_optimizers(self):
        lr = checklist(self.config.training.get('lr', 1e-4), n=2)
        if self.config.training.mode == "wasserstein":
            betas = (0.5, 0.999)
        else:
            betas = (0.9, 0.999)
        gen_optim = torch.optim.Adam(self.generator.parameters(), lr=lr[0] or 1e-4, betas=betas)
        disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr[1] or 1e-4, betas=betas)
        self.config.training.balance = self.config.training.get('balance')
        return gen_optim, disc_optim

    def sample_prior(self, batch=None, shape=None, y=None):
        if batch is not None:
            batch_len = len(batch.shape) - len(self.generator.target_shape)
            batch_shape = batch.shape[:batch_len]
        else:
            batch_shape = shape
        z = self.prior((*batch_shape, *self.generator.input_shape)).sample()
        return z.to(self.device)

    def full_forward(self, batch, batch_idx=None, trace=None, sample=True):
        batch = batch.to(self.device)
        batch_len = len(batch.shape) - len(self.generator.target_shape)
        z = self.prior((*batch.shape[:batch_len], *checklist(self.config.latent.dim))).sample()
        x = self.generator(z.to(self.device))
        y_fake = self.discriminator(x)
        y_true = self.discriminator(batch)
        return z, x, (y_fake, y_true)

    def get_labels(self, out, phase):
        margin = float(self.config.training.get('margin', 0))
        invert = bool(self.config.training.get('invert_labels', False))
        if phase == 0:
            # get generator lables
            labels = torch.ones((*out.shape[:-1], 1), device=self.device) - margin
            if invert:
                labels = 1 - labels
            return labels
        elif phase == 1:
            true_labels = torch.ones((*out.shape[:-1], 1), device=self.device) - margin
            fake_labels = torch.zeros((*out.shape[:-1], 1), device=self.device)
            if invert:
                true_labels = 1 - true_labels
                fake_labels = 1 - fake_labels
            return true_labels, fake_labels

    def step(self, gen_loss, disc_loss):
        g_opt, d_opt = self.optimizers()
        if self._loss_phase == 0 or self._loss_phase is None:
            g_opt.zero_grad()
            self.manual_backward(gen_loss)
            g_opt.step()
        if self._loss_phase == 1 or self._loss_phase is None:
            d_opt.zero_grad()
            self.manual_backward(disc_loss)
            d_opt.step()

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

    def generate(self, x, z, trace=None):
        return self.generator(z, trace=trace)

    def decode(self, z=None):
        #TODO better batch shape handling (quick fix for NIME)
        z = z.reshape(z.shape[0], *self.generator.input_shape)
        return self.generate(None, z)

    def discriminate(self, x, z, return_hidden=False, trace=None):
        if isinstance(x, dist.Distribution):
            x = x.rsample()
        if self.input_augmentations is not None:
            x = self.input_augmentations(x)
        return self.discriminator(x, return_hidden=return_hidden, trace=trace)

    def discriminator_loss(self, discriminator, batch, generation, d_real, d_fake, drop_detail=False, batch_idx=0):
        if self.config.training.mode in ["logistic", "logistic_ns"]:
            labels_real, labels_fake = self.get_labels(d_real, phase=1)
            true_loss = nn.functional.binary_cross_entropy(d_real, labels_real)
            fake_loss = nn.functional.binary_cross_entropy(d_fake, labels_fake)
            disc_loss = (true_loss + fake_loss) 
        elif self.config.training.mode in ["softplus", "softplus_ns"]:
            true_loss = F.softplus(-d_real).mean()
            fake_loss = F.softplus(d_fake).mean()
            disc_loss = (true_loss + fake_loss) 
        elif self.config.training.mode == "hinge":
            true_loss = torch.relu(1 - d_real).mean()
            fake_loss = torch.relu(1 + d_fake).mean()
            disc_loss = (true_loss + fake_loss) / 2
        elif self.config.training.mode == "wasserstein":
            disc_loss = - d_real.mean() + d_fake.mean()
        losses = {'disc_loss': disc_loss.cpu().detach()}
        if self.config.training.get('reg_period', True):
            if batch_idx % self.config.training.get('reg_period', 1) == 0:
                if self.config.training.get('wdiv'):
                    if d_real.grad_fn is not None:
                        p = self.config.training.get('wdiv_exp', 6)
                        weight = float(self.config.training.wdiv)
                        labels_real, labels_fake = self.get_labels(d_real, phase=1)
                        fake_grad = torch.autograd.grad(d_fake, generation, labels_fake, create_graph=True, retain_graph=True,
                                                        only_inputs=True)[0]
                        real_grad = \
                        torch.autograd.grad(d_real, batch, labels_real, create_graph=True, retain_graph=True, only_inputs=True)[
                            0]
                        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
                        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
                        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * weight / 2
                        losses['wdiv'] = div_gp.cpu().detach()
                        disc_loss = disc_loss + div_gp
                if self.config.training.get('gp'):
                    if d_real.grad_fn is not None:
                        weight = float(self.config.training.gp)
                        gradient_penalty = self.compute_gradient_penalty(batch, generation)
                        disc_loss = disc_loss + weight * gradient_penalty
                        losses['gradient_penalty'] = gradient_penalty.cpu().detach()
                if self.config.training.get('r1'):
                    if d_real.grad_fn is not None:
                        weight = float(self.config.training.r1) / 2
                        labels, _ = self.get_labels(d_real, phase=1)
                        r1_reg = self.gradient_regularization(batch, labels)
                        losses['r1'] = r1_reg.cpu().detach()
                        disc_loss = disc_loss + weight * r1_reg
                if self.config.training.get('r2'):
                    if d_real.grad_fn is not None:
                        weight = float(self.config.training.r2) / 2
                        _, labels = self.get_labels(generation, phase=1)
                        r2_reg = self.gradient_regularization(generation)
                        losses['r2'] = r2_reg.cpu().detach()
                        disc_loss = disc_loss + weight * r2_reg
        if drop_detail:
            return disc_loss, losses
        else:
            return disc_loss

    def regularize_discriminator(self, discriminator, d_real, d_fake, d_loss):
        if self.config.training.mode == "wasserstein":
            clip_value = self.config.training.get('wgan_clip')
            if clip_value is not None:
                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

    def generator_loss(self, generator, batch, out, d_fake, z, hidden=None, drop_detail=False, batch_idx=0):
        if self.config.training.mode == "logistic":
            labels_real = self.get_labels(d_fake, phase=0)
            gen_loss = nn.functional.binary_cross_entropy(d_fake, labels_real)
        elif self.config.training.mode == "softplus":
            gen_loss = -F.softplus(d_fake).mean()
        elif self.config.training.mode == "softplus_ns":
            gen_loss = F.softplus(-d_fake).mean()
        elif self.config.training.mode == "hinge":
            gen_loss = -d_fake.mean()
        elif self.config.training.mode == "wasserstein":
            gen_loss = -d_fake.mean()
        losses = {'gen_loss': gen_loss.detach().cpu()}
        if self.config.training.get('reg_period', True):
            if batch_idx % self.config.training.get('reg_period', 1) == 0:
                if self.config.training.get('feature_matching'):
                    fm_loss = 0.
                    for i in range(len(hidden[0]) - 1):
                        feature_dim = len(hidden[0][i].shape) - len(d_fake.shape[:-1])
                        sum_dims = tuple(range(0, -feature_dim, -1))
                        # fm_loss = fm_loss + (hidden[0][i] - hidden[1][i]).pow(2).sum(sum_dims).sqrt().mean()
                        fm_loss = fm_loss + (hidden[0][i] - hidden[1][i]).abs().mean()
                    gen_loss = gen_loss + float(self.config.training['feature_matching']) * fm_loss
                    losses['feature_matching'] = fm_loss.detach().cpu()
        if drop_detail:
            return gen_loss, losses
        else:
            return gen_loss

    def training_step(self, batch, batch_idx):
        batch, y = batch
        batch = batch.to(self.device)
        g_opt, d_opt = self.optimizers()
        g_opt.zero_grad()
        d_opt.zero_grad()
        # generate
        z = self.sample_prior(batch=batch)
        out = self.generate(batch, z)
        # update discriminator
        g_losses = d_losses = None
        loss = 0
        if self._loss_phase == 0 or self._loss_phase is None:
            out_disc = (out if not isinstance(out, dist.Distribution) else out.sample()).detach()
            if self.config.training.get('wdiv'):
                batch.requires_grad_(True)
                out_disc.requires_grad_(True)
            d_real = self.discriminate(batch, z)
            d_fake = self.discriminate(out_disc, z)
            d_loss, d_losses = self.discriminator_loss(self.discriminator, batch, out_disc, d_real, d_fake, drop_detail=True, batch_idx=batch_idx)
            loss = loss + d_loss
            self.manual_backward(d_loss)
            d_opt.step()
            self.regularize_discriminator(self.discriminator, d_real, d_fake, d_loss)
        # update generator
        if self._loss_phase == 1 or self._loss_phase is None:
            if self.config.training.get('feature_matching'):
                d_fake, hidden_fake = self.discriminate(out, z, return_hidden=True)
                d_true, hidden_true = self.discriminate(batch, z, return_hidden=True)
                g_loss, g_losses = self.generator_loss(self.generator, batch, out, d_fake, z, hidden=[hidden_true, hidden_fake], drop_detail=True, batch_idx=batch_idx)
            else:
                d_fake = self.discriminate(out, z, return_hidden=False)
                g_loss, g_losses = self.generator_loss(self.generator, batch, out, d_fake, z, drop_detail=True, batch_idx=batch_idx)
            loss = loss + g_loss
            self.manual_backward(g_loss)
            g_opt.step()
        if g_losses is not None:
            self.log_losses(g_losses, "train")
        if d_losses is not None:
            self.log_losses(d_losses, "train")
        self.log("loss/train", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        batch, y = batch
        batch = batch.to(self.device)
        # generate
        z = self.sample_prior(batch=batch)
        out = self.generate(batch, z)
        # get discriminator loss
        g_loss = d_loss = None
        d_real, hidden_real = self.discriminate(batch, z, return_hidden=True)
        d_fake, hidden_fake = self.discriminate(out, z, return_hidden=True)
        d_loss, d_losses = self.discriminator_loss(self.discriminator, batch, out, d_real, d_fake, drop_detail=True)
        # get generator loss
        g_loss, g_losses = self.generator_loss(self.generator, batch, out, d_fake, z, hidden=[hidden_real, hidden_fake], drop_detail=True)
        self.log_losses(g_losses, "valid")
        self.log_losses(d_losses, "valid")
        loss = (g_loss or 0) + (d_loss or 0)
        self.log("loss/valid", loss, prog_bar=False)
        return loss

    # External methods
    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        temperature = checklist(temperature)
        generations = []
        for t in temperature:
            z = torch.randn((n_samples, *self.generator.input_shape), device=self.device) * t
            x = self.generator(z)
            if isinstance(x, dist.Distribution):
                if sample:
                    x = x.sample()
                else:
                    x = x.mean
            generations.append(x)
        return torch.stack(generations, 1)

    def gradient_regularization(self, inputs, labels):
        inputs.requires_grad_(True)
        d_out = self.discriminate(inputs, None)
        gradients = \
        torch.autograd.grad(outputs=d_out, inputs=inputs, grad_outputs=labels, allow_unused=True, create_graph=True,
                            retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        data_shape = len(real_samples.shape[1:])
        alpha = torch.rand((*real_samples.shape[:-data_shape], *(1,) * data_shape), device=real_samples.device)
        # Get random interpolation between real and fake samples
        if isinstance(fake_samples, dist.Distribution):
            fake_samples = fake_samples.sample()
        interpolates = (alpha * real_samples.detach() + ((1 - alpha) * fake_samples.detach()))
        interpolates.requires_grad_(True)
        d_interpolates = self.discriminate(interpolates, None)
        fake = torch.full((real_samples.shape[0], 1), 1.0, device=real_samples.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def trace(self, x: torch.Tensor, sample: bool = False):
        if isinstance(x, (tuple, list)):
            x, y = x
        z = self.sample_prior(x)
        trace_model = {'generator': {}, 'discriminator' : {}}
        out = self.generate(x, z, trace=trace_model['generator'])
        _ = self.discriminate(out, z, trace=trace_model['discriminator'])
        trace = {}
        trace['histograms'] = trace_model
        return trace


class ProgressiveGAN(GAN):
    def __init__(self, config=None) -> None:

        super().__init__(config=config)
        self.config.training.training_schedule = checklist(self.config.training.get('training_schedule'),
                                                           n=len(self.generator))
        assert len(self.config.training.training_schedule) == len(self.generator) - 1
        self.config.training.transition_schedule = checklist(self.config.training.get('transition_schedule'),
                                                             n=len(self.generator) - 1)
        assert len(self.config.training.transition_schedule) == len(self.generator) - 1
        # self.init_rgb_modules()
        self.save_hyperparameters(dict(self.config))
        self._current_phase = None
        self._transition = None

    def init_generator(self, config: OmegaConf) -> None:
        generator_type = config.type or "DeconvEncoder"
        config.args.reshape_method="pgan"
        self.generator = getattr(encoders, generator_type)(config.args)

    def init_discriminator(self, config: OmegaConf) -> None:
        disc_type = config.type or "ConvEncoder"
        config.args.target_shape = 1
        config.args.reshape_method="pgan"
        self.discriminator = getattr(encoders, disc_type)(config.args)

    def configure_optimizers(self):
        lr = checklist(self.config.training.get('lr', 1e-4), n=2)
        gen_optim = torch.optim.Adam(self.generator.parameters(), lr=lr[0] or 1e-4)
        disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr[1] or 1e-4)
        self.config.training.balance = self.config.training.get('balance')
        return gen_optim, disc_optim

    def _get_mixing_factor(self):
        if self._phase_counter > self.config.training.transition_schedule[self._current_phase - 1]:
            return None
        if self._current_phase == 0:
            return None
        alpha = min(1, float(self._phase_counter / self.config.training.transition_schedule[self._current_phase-1]))
        return alpha

    def generate(self, x, z, trace=None, transition=True):
        if transition:
            alpha = self._get_mixing_factor()
            generation = self.current_generator(z, transition=alpha, trace=trace)
        else:
            generation = self.current_generator(z, trace=trace)
        return generation

    def discriminate(self, x, z, trace=None, return_hidden=False):
        if self._transition == 1:
            alpha = self._get_mixing_factor()
            return self.current_discriminator(x, trace=trace, return_hidden=return_hidden, transition=alpha)
        else:
            return self.current_discriminator(x, trace=trace, return_hidden=return_hidden)

    def _get_sub_generator(self, phase):
        generator = self.generator[:phase + 1]
        return generator

    def _get_sub_discriminator(self, phase):
        return self.discriminator[-(phase + 1):]

    def _get_generator_upsample(self, module):
        ds_module = None
        if hasattr(module, "upsample"):
            ds_module = module.upsample
        elif isinstance(module, nn.Sequential):
            for m in module:
                ds_module = m.__dict__['_modules'].get('upsample')
                if ds_module is not None:
                    break
        return ds_module

    def _get_discriminator_downsample(self, module):
        ds_module = None
        if hasattr(module, "downsample"):
            ds_module = module.downsample
        elif hasattr(module, "conv_modules"):
            for m in module.conv_modules:
                ds_module = m.__dict__['_modules'].get('downsample')
                if ds_module is not None:
                    break
        return ds_module

    def _get_discriminator_downsamples(self, phase):
        downsamples = []
        for i in range(len(self.discriminator) - (phase + 1)):
            ds_module = self._get_discriminator_downsample(self.discriminator[i])
            if ds_module is not None:
                downsamples.append(ds_module)
        return nn.Sequential(*downsamples)

    def training_step(self, batch, batch_idx):
        batch, y = batch
        batch = batch.to(self.device)
        g_opt, d_opt = self.optimizers()
        # generate
        z = self.sample_prior(batch=batch)
        out = self.generate(batch, z)
        # downsample image in case
        if self._current_phase is not None and self._current_phase != len(self.config.training.training_schedule):
            batch = self._get_discriminator_downsamples(self._current_phase)(batch)

        g_loss = d_loss = g_losses = d_losses = None
        if self._loss_phase == 0 or self._loss_phase is None:
            d_real = self.discriminate(batch, z)
            d_fake = self.discriminate(out.detach(), z)
            d_loss, d_losses = self.discriminator_loss(self.current_discriminator, batch, out, d_real, d_fake, drop_detail=True, batch_idx=batch_idx)
            d_opt.zero_grad()
            self.manual_backward(d_loss)
            d_opt.step()
            self.regularize_discriminator(self.current_discriminator, d_real, d_fake, d_loss)
        # update generator
        if self.config.training.get('reg_period', True):
            if batch_idx % self.config.training.get('reg_period', 1) == 0:
                d_fake, hidden_fake = self.discriminate(out, z, return_hidden=True)
                d_true, hidden_true = self.discriminate(batch, z, return_hidden=True)
                g_loss, g_losses = self.generator_loss(self.generator, batch, out, d_fake, z, hidden=[hidden_true, hidden_fake], drop_detail=True, batch_idx=batch_idx)
            else:
                d_fake = self.discriminate(out, z, return_hidden=False)
                g_loss, g_losses = self.generator_loss(self.generator, batch, out, d_fake, z, drop_detail=True, batch_idx=batch_idx)
            g_opt.zero_grad()
            self.manual_backward(g_loss)
            g_opt.step()
        if g_losses is not None:
            self.log_losses(g_losses, "train")
        if d_loss is not None:
            self.log_losses(d_losses, "train")
        loss = (g_loss or 0) + (d_loss or 0)
        self.log("loss/train", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        batch, y = batch
        batch = batch.to(self.device)
        # generate
        z = self.sample_prior(batch=batch)
        out = self.generate(batch, z)
        # get discriminator loss
        if self._current_phase is not None and self._current_phase != len(self.config.training.training_schedule):
            batch = self._get_discriminator_downsamples(self._current_phase)(batch)
        g_losses = d_losses = None
        d_real, hidden_real = self.discriminate(batch, z, return_hidden=True)
        d_fake, hidden_fake = self.discriminate(out, z, return_hidden=True)
        d_loss, d_losses = self.discriminator_loss(self.current_discriminator, batch, out, d_real, d_fake, drop_detail=True, batch_idx=batch_idx)
        # get generator loss
        g_loss, g_losses = self.generator_loss(self.current_generator, batch, out, d_fake, z, hidden=[hidden_real, hidden_fake], drop_detail=True, batch_idx=batch_idx)
        if g_losses is not None:
            self.log_losses(g_losses, "valid")
        if d_losses is not None:
            self.log_losses(d_losses, "valid")
        loss = (g_loss or 0) + (d_loss or 0)
        self.log("loss/valid", loss, prog_bar=False)
        return loss

    def on_train_start(self):
        self._phase_counter = 0
        self._current_phase = self.config.training.get('phase_at_init', 0)
        # no transition at first epoch
        if self._current_phase == 0:
            self._transition = self.config.training.get('transition_at_init', 0)
            self.current_generator = self._get_sub_generator(self._current_phase)
            self.current_discriminator = self._get_sub_discriminator(self._current_phase)
        else:
            self._transition = self.config.training.get('transition_at_init', 1)
            if self._transition:
                self.current_generator = self._get_sub_generator(self._current_phase - 1)
                self.current_discriminator = self._get_sub_discriminator(self._current_phase - 1)
            else:
                self.current_generator = self._get_sub_generator(self._current_phase)
                self.current_discriminator = self._get_sub_discriminator(self._current_phase)

    def on_validation_start(self):
        if self._current_phase is None:
            self.on_train_start()

    def on_train_epoch_start(self):
        super(ProgressiveGAN, self).on_train_epoch_start()
        # check learning phase
        if self._current_phase != len(self.config.training.training_schedule):
            if self._phase_counter >= self.config.training.training_schedule[self._current_phase]:
                self._current_phase += 1
                self._transition = 1
                self._phase_counter = 0
                if self._current_phase == len(self.generator):
                    self.current_generator = self.generator
                    self.current_discriminator = self.discriminator
                else:
                    self.current_generator = self._get_sub_generator(self._current_phase)
                    self.current_discriminator = self._get_sub_discriminator(self._current_phase)


    def on_train_epoch_end(self):
        self._phase_counter += 1
        self.training_epoch += 1

    # External methods
    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        temperature = checklist(temperature)
        generations = []
        for t in temperature:
            z = self.prior((n_samples, *self.generator.input_shape), device=self.device).sample() * t
            if self._current_phase is not None:
                x = self._get_sub_generator(self._current_phase)(z)
            else:
                x = self.generator(z)
            if isinstance(x, dist.Distribution):
                if sample:
                    x = x.sample()
                else:
                    x = x.mean
            generations.append(x)
        return torch.stack(generations, 1)


class ModulatedGAN(ProgressiveGAN):
    def __init__(self, config=None, encoder=None, **kwargs) -> None:
        # generator input is not the latent vector, but a constant
        # config.generator.args.input_shape = config.generator.args.channels[-1]
        config.generator.args.reshape_method="none"
        super().__init__(config=config, **kwargs)
        # build encode
        self.config.encoder = self.config.get('encoder') or encoder
        self.config.encoder['mode'] = self.config.encoder.get('mode', 'sequential')
        self.init_encoder(self.config.encoder)
        self.config.training.style_mixing_prob = self.config.training.get('style_mixing_prob', 0.9)
        # build constant input
        self.const = nn.Parameter(torch.randn(self.generator.input_shape))
        self.config.training.path_length_decay = self.config.training.get('path_length_decay', 1e-2)
        self.register_buffer("path_means", torch.tensor(torch.nan))
        self.save_hyperparameters(dict(self.config))

    def init_encoder(self, encoder_config):
        encoder_args = encoder_config.get('args', {})
        if encoder_config.mode == "sequential":
            encoder_args['input_shape'] = self.config.latent.get('dim', 512)
            encoder_args['target_shape'] = encoder_args.get('target_shape', 512)
            self.encoder = getattr(encoders, encoder_config.get('type', 'MLPEncoder'))(encoder_args)
        else:
            dims = self.generator.channels
            encoder_list = []
            for d in dims:
                current_args = dict(encoder_args)
                current_args.target_shape = d
                encoder_list.append(getattr(encoders, encoder_config.get('type', 'MLPEncoder'))(encoder_args))
            self.encoder = nn.ModuleList(encoder_list)

    def init_generator(self, config: OmegaConf) -> None:
        generator_type = config.type or "DeconvEncoder"
        config.args.input_shape = None
        config.args.reshape_method="none"
        config.args.layer = config.args.get('layer', 'ModUpsamplingBlock')
        self.generator = getattr(encoders, generator_type)(config.args)

    def init_discriminator(self, config: OmegaConf) -> None:
        disc_type = config.type or "ConvEncoder"
        config.args.target_shape = 1
        config.args.reshape_method="flatten"
        config.args.index_flatten = False
        config.args.flatten_args = {'nlayers': 0, 'norm':0, 'nnlin':None}
        self.discriminator = getattr(encoders, disc_type)(config.args)

    def sample_prior(self, batch=None, shape=None):
        if batch is not None:
            batch_len = len(batch.shape) - len(self.generator.target_shape)
            z = self.prior((*batch.shape[:batch_len], *self.encoder.input_shape)).sample()
        else:
            z = self.prior(*shape, *self.encoder.input_shape).sample()
        return z.to(self.device)

    def get_modulations(self, z, trace=None):
        if self.config.encoder.mode == "sequential":
            if isinstance(z, tuple):
                styles = tuple([self.encoder(z_tmp) for z_tmp in z])
                if trace is not None:
                    trace['styles'] = torch.stack(styles)
            else:
                styles = self.encoder(z)
                if trace is not None:
                    trace['styles'] = styles
        else:
            styles = [enc(z) for enc in self.encoder]
            if trace is not None:
                for i, s in enumerate(styles):
                    trace['styles_%d'%i] = s
        return styles

    def get_const(self, z):
        if isinstance(z, tuple):
            z = z[0]
        const = self.const.view(*(1,) * (z.ndim - 1), *self.const.shape)
        return const.repeat(*z.shape[:-1], *(1,) * (self.generator.dim + 1))

    def sample_prior(self, batch=None, shape=None, mixing_prob=None):
        mixing_prob = self.config.training.style_mixing_prob if mixing_prob is None else mixing_prob
        if mixing_prob and len(self.current_generator) > 1:
            if random.random() < mixing_prob:
                return (self.sample_prior(batch, shape, 0), self.sample_prior(batch, shape, 0))
        if batch is not None:
            batch_len = len(batch.shape) - len(self.generator.target_shape)
            z = self.prior((*batch.shape[:batch_len], *self.encoder.input_shape)).sample()
        else:
            z = self.prior(*shape, *self.encoder.input_shape).sample()
        return z.to(self.device)

    def generate(self, batch, z, eps=None, trace=None):
        const = self.get_const(z)
        if isinstance(z, tuple) and len(self.current_generator) > 1:
            layer_range = list(range(len(self.current_generator)))
            crossover_idx = random.randrange(0, len(layer_range)-1)
            y1, y2 = self.get_modulations(z, trace=trace)
            y = [y1] * crossover_idx + [y2] * (len(layer_range) - crossover_idx)
            y = tuple(y)
        else:
            y = self.get_modulations(z, trace=trace)
        if self._transition == 1:
            alpha = self._get_mixing_factor()
            generation = self.current_generator(const, mod=y, transition=alpha, trace=trace)
        else:
            generation = self.current_generator(const, mod=y, trace=trace)
            # if self._current_phase is not None and (self._current_phase != len(self.config.training.training_schedule)):
            #    generation = self.generator.toRGB_modules[self._current_phase](generation)
        return generation

    def generator_loss(self, generator, batch, out, d_fake, z, hidden=None, drop_detail=False, batch_idx=0):
        g_loss = super(ModulatedGAN, self).generator_loss(generator, batch, out, d_fake, z, hidden=hidden, drop_detail=drop_detail)
        g_losses = None
        if drop_detail:
            g_loss, g_losses = g_loss
        if self.config.training.get('reg_period', True):
            if batch_idx % self.config.training.get('reg_period', 1) == 0:
                if self.config.training.get('path_length') and g_loss.grad_fn is not None:
                    n_batch = self.config.training.get('path_length_batches', 64)
                    n_samples = self.config.training.get('path_length_samples', 3)
                    path_length_penalty = self.path_length_penalty(n_batch=n_batch, n_samples=n_samples)
                    g_loss = g_loss + float(self.config.training.path_length) * path_length_penalty
                    if g_losses is not None:
                        g_losses['path_length'] = path_length_penalty.cpu().detach()
        if drop_detail:
            return g_loss, g_losses
        else:
            return g_loss

    def path_length_penalty(self, n_batch=64, n_samples=4):
        z = torch.randn((n_batch * n_samples, *checklist(self.config.latent.dim)), device=self.device,
                        requires_grad=True)
        y = self.get_modulations(z)
        const = self.get_const(z)
        generations = self.current_generator(const, mod=y)
        noise = torch.randn_like(generations) * np.sqrt(np.prod(generations.shape[-(self.generator.dim + 1):]))
        corrupted = (generations * noise).sum()
        gradients = torch.autograd.grad(outputs=corrupted, inputs=y, create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(n_batch, n_samples, gradients.shape[-1])
        path_lengths = gradients.pow(2).sum(-1).mean(-1).sqrt()
        if self.path_means.isnan():
            self.path_means = path_lengths.mean()
            return torch.tensor(0., dtype=z.dtype, device=self.device)
        else:
            self.path_means = (self.path_means + self.config.training.path_length_decay * (path_lengths.mean() - self.path_means)).detach()
            pl_penalty = (path_lengths - self.path_means).pow(2).mean()
            return pl_penalty

    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        temperature = checklist(temperature)
        generations = []
        for t in temperature:
            z = torch.randn((n_samples, *checklist(self.config.latent.dim)), device=self.device) * t
            if self._current_phase is not None:
                x = self._get_sub_generator(self._current_phase)(self.get_const(z), mod=z)
            else:
                x = self.generator(z)
            if isinstance(x, dist.Distribution):
                if sample:
                    x = x.sample()
                else:
                    x = x.mean
            generations.append(x)
        return torch.stack(generations, 1)

    def trace(self, x: torch.Tensor, sample: bool = False):
        if isinstance(x, (tuple, list)):
            x, y = x
        z = self.sample_prior(x, mixing_prob=0)
        trace_model = {'generator': {}, 'discriminator' : {}}
        out = self.generate(x, z, trace=trace_model['generator'])
        _ = self.discriminate(out, z, trace=trace_model['discriminator'])
        trace = {}
        trace['histograms'] = trace_model
        return trace


