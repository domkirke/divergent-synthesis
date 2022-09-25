from omegaconf import OmegaConf
import torch, torch.optim as optim, torch.nn as nn, torch.nn.functional as F
from .. import generators, discriminators
from ..modules import priors
from .model import Model
import torch.distributions as dist 
from ..utils import checklist, checksize


class GAN(Model):
    gan_modes = ['logistic', 'hinge', 'wasserstein', 'softplus', 'softplus-ns']
    def __init__(self, config: OmegaConf = None):
        # manage config
        config = config or OmegaConf.create()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        config_checkpoint = config.get('config_checkpoint')
        super().__init__(config=config)

        config.latent = config.get('latent')
        self.prior = getattr(priors, config.latent.get('prior', "isotropic_gaussian"))

        generator_class = config.generator.type
        self.generator = getattr(generators, generator_class)(config)
        discriminator_class = config.discriminator.type
        self.discriminator = getattr(discriminators, discriminator_class)(config)

        # setup training
        config.training = config.get('training')
        config.training.mode = config.training.get('mode', 'logistic')
        assert config.training.mode in self.gan_modes
        self.automatic_optimization = False
        # self.reconstruction_losses = parse_additional_losses(config.training.get('rec_losses'))
        # self.training_epoch = config.get('training_epoch', 0)
        # # load from checkpoint
        # if config_checkpoint:
        #     self.import_checkpoint(config_checkpoint)
        self.config = config
    
    ## ______________________________________________________________________________________
    ## Lightning handlers

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
        out = self.generate(x=x, z=z, y=y)
        g_losses = d_losses = None
        loss = 0
        if self._loss_phase == 0 or self._loss_phase is None:
            # update discriminator
            d_loss, d_losses = self.optimize_discriminator(out, x=x, y=y, z=z, batch_idx=batch_idx)
            loss = loss + d_loss
        if self._loss_phase == 1 or self._loss_phase is None:
            # update generator
            g_loss, g_losses = self.optimize_generator(out, x=x, y=y, z=z, batch_idx=batch_idx)
            loss = loss + g_loss
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
        out = self.generate(x=x, y=y, z=z)
        # get discriminator loss
        g_loss = d_loss = None
        d_real, hidden_real = self.discriminate(x=x, y=y, z=z, return_hidden=True)
        d_fake, hidden_fake = self.discriminate(x=out, y=y, z=z, return_hidden=True)
        d_loss, d_losses = self.discriminator_loss(self.discriminator, x, out, d_real, d_fake, z=z, y=y, drop_detail=True)
        # get generator loss
        g_loss, g_losses = self.generator_loss(self.generator, x, out, d_fake, z=z, y=y, hidden=[hidden_real, hidden_fake], drop_detail=True)
        self.log_losses(g_losses, "valid")
        self.log_losses(d_losses, "valid")
        loss = (g_loss or 0) + (d_loss or 0)
        self.log("loss/valid", loss, prog_bar=False)
        return loss


    ## ______________________________________________________________________________________
    ## High-level functions 

    def sample_prior(self, batch=None, shape=None, y=None):
        """Samples the GAN's prior to obtain latent positions."""
        if batch is not None:
            batch_len = len(batch.shape) - len(self.generator.target_shape)
            batch_shape = batch.shape[:batch_len]
        else:
            batch_shape = shape
        y = None
        z = self.prior((*batch_shape, *checksize(self.config.latent.dim))).sample()
        return {'z': z.to(self.device), 'y':y}

    def trace(self, x: torch.Tensor, sample: bool = False):
        if isinstance(x, (tuple, list)):
            x, y = x
        z = self.sample_prior(x)
        trace_model = {'generator': {}, 'discriminator' : {}}
        out = self.generate(x=x, y=z['y'], z=z['z'], trace=trace_model['generator'])
        _ = self.discriminate(x=out, y=z['y'], z=z['z'], trace=trace_model['discriminator'])
        trace = {}
        trace['histograms'] = trace_model
        return trace

    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        """Generates data by sampling the GAN's prior and decoding the correspondig data"""
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

    def generate(self, x=None, y=None, z=None, trace=None):
        return self.generator(z=z, y=y, trace=trace)

    def decode(self, z=None):
        #TODO better batch shape handling (quick fix for NIME)
        z = z.reshape(z.shape[0], *self.generator.input_shape)
        return self.generate(None, z)

    def discriminate(self, x=None, y=None, z=None, return_hidden=False, trace=None):
        if isinstance(x, dist.Distribution):
            x = x.rsample()
        # if self.input_augmentations is not None:
        #     x = self.input_augmentations(x)
        return self.discriminator(x, y=y, return_hidden=return_hidden, trace=trace)

    def full_forward(self, batch, batch_idx=None, trace=None, sample=True):
        # batch = batch.to(self.device)
        batch_len = len(batch.shape) - len(self.generator.target_shape)
        z = self.prior((*batch.shape[:batch_len], *checklist(self.config.latent.dim))).sample()
        x = self.generator(z)
        y_fake = self.discriminator(x)
        y_true = self.discriminator(batch)
        return z, x, (y_fake, y_true)

    def optimize_discriminator(self, out, x=None, y=None, z=None, batch_idx=None):
        d_opt = self.optimizers()[1]
        # assert z is not None
        out_disc = (out if not isinstance(out, dist.Distribution) else out.sample()).detach()
        if self.config.training.get('wdiv'):
            x.requires_grad_(True)
            out_disc.requires_grad_(True)
        d_real = self.discriminate(x=x, y=y, z=z)
        d_fake = self.discriminate(x=out_disc, y=y, z=z)
        d_loss, d_losses = self.discriminator_loss(self.discriminator, x, out_disc, d_real, d_fake, y=y, drop_detail=True, batch_idx=batch_idx)
        self.manual_backward(d_loss)
        d_opt.step()
        self.regularize_discriminator_after_update(self.discriminator, d_real, d_fake, d_loss)
        return d_loss, d_losses

    def optimize_generator(self, out, x=None, y=None, z=None, batch_idx=None):
        g_opt = self.optimizers()[0]
        if self.config.training.get('feature_matching'):
            d_true, hidden_true = self.discriminate(x=x, y=y, z=z, return_hidden=True)
            d_fake, hidden_fake = self.discriminate(x=out, y=y, z=z, return_hidden=True)
            g_loss, g_losses = self.generator_loss(self.generator, x, out, d_fake, z=z, hidden=[hidden_true, hidden_fake], y=y, drop_detail=True, batch_idx=batch_idx)
        else:
            d_fake = self.discriminate(out, z, return_hidden=False)
            g_loss, g_losses = self.generator_loss(self.generator, x, out, d_fake, z=z, y=y, drop_detail=True, batch_idx=batch_idx)
        self.manual_backward(g_loss)
        g_opt.step()
        return g_loss, g_losses

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


    ## ______________________________________________________________________________________
    ##  Discriminator loss

    def discriminator_loss(self, discriminator, batch, generation, d_real, d_fake, y=None, z=None, drop_detail=False, batch_idx=0):
        if self.config.training.mode in ["logistic", "logistic_ns"]:
            labels_real, labels_fake = self.get_labels(d_real, phase=1)
            d_real = torch.sigmoid(d_real['domain'])
            d_fake = torch.sigmoid(d_fake['domain'])
            true_loss = nn.functional.binary_cross_entropy(d_real['domain'], labels_real)
            fake_loss = nn.functional.binary_cross_entropy(d_fake['domain'], labels_fake)
            disc_loss = (true_loss + fake_loss) 
        elif self.config.training.mode in ["softplus", "softplus_ns"]:
            true_loss = F.softplus(-d_real['domain']).mean()
            fake_loss = F.softplus(d_fake['domain']).mean()
            disc_loss = (true_loss + fake_loss) 
        elif self.config.training.mode == "hinge":
            true_loss = torch.relu(1 - d_real['domain']).mean()
            fake_loss = torch.relu(1 + d_fake['domain']).mean()
            disc_loss = (true_loss + fake_loss) / 2
        elif self.config.training.mode == "wasserstein":
            disc_loss = - d_real['domain'].mean() + d_fake['domain'].mean()
        losses = {'disc_loss': disc_loss.cpu().detach()}

        # perform regularization
        if self.config.training.get('reg_period', True):
            if batch_idx % self.config.training.get('reg_period', 1) == 0:
                if self.config.training.get('wdiv'):
                    if d_real.grad_fn is not None:
                        p = self.config.training.get('wdiv_exp', 6)
                        weight = float(self.config.training.wdiv)
                        labels_real, labels_fake = self.get_labels(d_real['domain'], phase=1)
                        fake_grad = torch.autograd.grad(d_fake, generation, labels_fake, create_graph=True, retain_graph=True,
                                                        only_inputs=True)[0]
                        real_grad = \
                        torch.autograd.grad(d_real['domain'], batch, labels_real, create_graph=True, retain_graph=True, only_inputs=True)[
                            0]
                        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
                        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
                        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * weight / 2
                        losses['wdiv'] = div_gp.cpu().detach()
                        disc_loss = disc_loss + div_gp
                if self.config.training.get('gp'):
                    if d_real['domain'].grad_fn is not None:
                        weight = float(self.config.training.gp)
                        gradient_penalty = self.compute_gradient_penalty(batch, generation)
                        disc_loss = disc_loss + weight * gradient_penalty
                        losses['gradient_penalty'] = gradient_penalty.cpu().detach()
                if self.config.training.get('r1'):
                    if d_real['domain'].grad_fn is not None:
                        weight = float(self.config.training.r1) / 2
                        labels, _ = self.get_labels(d_real, phase=1)
                        r1_reg = self.gradient_regularization(batch, labels)
                        losses['r1'] = r1_reg.cpu().detach()
                        disc_loss = disc_loss + weight * r1_reg
                if self.config.training.get('r2'):
                    if d_real['domain'].grad_fn is not None:
                        weight = float(self.config.training.r2) / 2
                        _, labels = self.get_labels(generation, phase=1)
                        r2_reg = self.gradient_regularization(generation)
                        losses['r2'] = r2_reg.cpu().detach()
                        disc_loss = disc_loss + weight * r2_reg
        if drop_detail:
            return disc_loss, losses
        else:
            return disc_loss

    def regularize_discriminator_after_update(self, discriminator, d_real, d_fake, d_loss):
        if self.config.training.mode == "wasserstein":
            clip_value = self.config.training.get('wgan_clip')
            if clip_value is not None:
                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

    ## ______________________________________________________________________________________
    ##  Generator loss

    def generator_loss(self, generator, batch, out, d_fake, y=None, z=None, hidden=None, drop_detail=False, batch_idx=0):
        if self.config.training.mode == "logistic":
            labels_real = self.get_labels(d_fake['domain'], phase=0)
            gen_loss = nn.functional.binary_cross_entropy(d_fake['domain'], labels_real)
        elif self.config.training.mode == "softplus":
            gen_loss = -F.softplus(d_fake['domain']).mean()
        elif self.config.training.mode == "softplus_ns":
            gen_loss = F.softplus(-d_fake['domain']).mean()
        elif self.config.training.mode == "hinge":
            gen_loss = -d_fake['domain'].mean()
        elif self.config.training.mode == "wasserstein":
            gen_loss = -d_fake['domain'].mean()
        losses = {'gen_loss': gen_loss.detach().cpu()}
        if self.config.training.get('reg_period', True):
            if batch_idx % self.config.training.get('reg_period', 1) == 0:
                if self.config.training.get('feature_matching'):
                    fm_loss = 0.
                    for i in range(len(hidden[0]) - 1):
                        feature_dim = len(hidden[0][i].shape) - len(d_fake['domain'].shape[:-1])
                        sum_dims = tuple(range(0, -feature_dim, -1))
                        # fm_loss = fm_loss + (hidden[0][i] - hidden[1][i]).pow(2).sum(sum_dims).sqrt().mean()
                        fm_loss = fm_loss + (hidden[0][i] - hidden[1][i]).abs().mean()
                    gen_loss = gen_loss + float(self.config.training['feature_matching']) * fm_loss
                    losses['feature_matching'] = fm_loss.detach().cpu()
        if drop_detail:
            return gen_loss, losses
        else:
            return gen_loss

    ## ______________________________________________________________________________________
    ##  Helper functions
    
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
        d_interpolates = self.discriminate(interpolates, None)['domain'].unsqueeze(1)
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

    def get_labels(self, out, phase):
        margin = float(self.config.training.get('margin', 0.))
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

    def log_losses(self, loss_dict, period, **kwargs):
        for k, v in loss_dict.items():
            self.log(f"{k}/{period}", v, **kwargs)


class ACGAN(GAN):

    def sample_prior(self, batch=None, shape=None, y=None):
        """Samples the GAN's prior to obtain latent positions."""
        if batch is not None:
            batch_len = len(batch.shape) - len(self.generator.target_shape)
            batch_shape = batch.shape[:batch_len]
        else:
            batch_shape = shape
        y = {}
        for task, task_params in self.config.classes.items():
            y[task] = torch.randint(0, task_params['dim'], batch_shape, device=self.device)
        z = self.prior((*batch_shape, *checksize(self.config.latent.dim))).sample()
        return {'z': z.to(self.device), 'y':y}

    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        """Generates data by sampling the GAN's prior and decoding the correspondig data"""
        temperature = checklist(temperature)
        generations = []
        for t in temperature:
            z = torch.randn((n_samples, self.config.latent.dim), device=self.device) * t
            y = {}
            for task, task_params in self.config.classes.items():
                y[task] = torch.randint(0, task_params['dim'], (n_samples,), device=self.device)
            x = self.generator(z, y=y)
            if isinstance(x, dist.Distribution):
                if sample:
                    x = x.sample()
                else:
                    x = x.mean
            generations.append(x)
        return torch.stack(generations, 1)

    def get_classification_loss(self, y=None, y_out=None, drop_detail=False):
        loss = 0; losses = {}
        for k, v in self.config.classes.items():
            if (k in y_out) and (k in y):
                if isinstance(y_out[k], dist.Distribution):
                    loss_tmp = -(y_out[k].log_prob(y[k]).sum(-1).mean())
                    loss = loss + loss_tmp
                    losses = {**losses, 'loss': loss_tmp.detach().item()}
                else:
                    raise NotImplementedError
        if drop_detail:
            return loss, losses
        else:
            return loss

    def discriminator_loss(self, discriminator, batch, generation, d_real, d_fake, z=None, y=None,  drop_detail=False, batch_idx=0):
        loss, losses = super(ACGAN, self).discriminator_loss(discriminator, batch, generation, d_real, d_fake, drop_detail=drop_detail, batch_idx=batch_idx)
        classif_loss_real, classif_losses_real = self.get_classification_loss(y=y, y_out = d_real, drop_detail=True)
        loss = loss + classif_loss_real
        classif_loss_fake, classif_losses_fake = self.get_classification_loss(y=y, y_out = d_fake, drop_detail=True)
        loss = loss + classif_loss_fake
        classif_losses = {**{f"{k}_fake": v for k, v in classif_losses_fake.items()},
                          **{f"{k}_real": v for k, v in classif_losses_real.items()}}
        if drop_detail:
            losses = {**classif_losses, **losses}
            return loss, losses
        else:
            return loss