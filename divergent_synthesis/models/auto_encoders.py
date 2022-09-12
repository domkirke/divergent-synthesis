import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, sys, pdb, re
sys.path.append('../')
from sklearn import decomposition 
from omegaconf import OmegaConf, ListConfig
from divergent_synthesis.models.model import Model, ConfigType
from divergent_synthesis.models.gans import GAN, parse_additional_losses
from divergent_synthesis.modules import encoders
from torch import distributions as dist
from divergent_synthesis.utils import checklist, trace_distribution, reshape_batch, flatten_batch
from divergent_synthesis.utils.misc import _recursive_to
from divergent_synthesis.losses import get_regularization_loss, get_distortion_loss, priors, regularization
from typing import Dict, Union, Tuple, Callable


class AutoEncoder(Model):
    def __init__(self, config: OmegaConf=None, **kwargs):
        """
        Base class for auto-encoding architectures. Configuration file must include:
        - encoder
        - decoder
        - latent
        - training
        - conditioning (optional)

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
        # latent configs
        config.latent = config.get('latent')
        self.latent = config.latent

        # conditioning parameters
        concat_embeddings = {}
        self.conditionings = {}
        if config.get('conditioning') is not None:
            assert config.conditioning.get('tasks') is not None, "conditioning need the tasks keyword argument"
            decoder_cat_conds = {}
            for i, t in enumerate(config.conditioning.tasks):
                config.conditioning.tasks[i]['target'] = checklist(config.conditioning.tasks[i].get('target', ['decoder']))
                config.conditioning.tasks[i]['mode'] = config.conditioning.tasks[i].get('mode', 'cat')
                if t.get('mode') == "cat":
                    concat_embeddings[t['name']] = OneHot(n_classes=t['dim'])
                    if "decoder" in t['target']:
                        decoder_cat_conds[t['name']] = t['dim']
                self.conditionings[t['name']] = t
            if config.conditioning.predictor and len(decoder_cat_conds) > 0:
                prediction_modules = {}
                for cond_name, cond_dim in decoder_cat_conds.items():
                    prediction_module_type = getattr(encoders, config.conditioning.predictor.get('type', 'MLPEncoder'))
                    config.conditioning.predictor.args.input_shape = config.latent.dim
                    config.conditioning.predictor.args.target_shape = cond_dim
                    prediction_modules[cond_name] = prediction_module_type(config.conditioning.predictor.get('args',{}))
                self.prediction_modules = nn.ModuleDict(prediction_modules)

        self.concat_embeddings = nn.ModuleDict(concat_embeddings)
            
        # encoder architecture
        config.encoder = config.get('encoder')
        config.encoder.args = config.encoder.get('args', {})
        if config.encoder['args'].get('input_shape') is None:
            config.encoder['args']['input_shape'] = self.get_encoder_shape(self.input_shape)
        if config.encoder['args'].get('target_shape') is None:
            config.encoder['args']['target_shape'] = config.latent.dim
        config.encoder['args']['target_dist'] = config.latent.dist
        encoder_type = config.encoder.type or "MLPEncoder"
        self.encoder = getattr(encoders, encoder_type)(config.encoder.args)

        # decoder architecture
        config.decoder = config.get('decoder')
        config.decoder.args = config.decoder.get('args', {})
        config.decoder.args.input_shape = self.get_decoder_shape(config.latent.dim)
        if config.decoder.args.get('input_shape') is None:
            config.decoder.args.input_shape = config.latent.dim
        if config.decoder.args.get('target_shape') is None:
            config.decoder.args.target_shape = self.input_shape
        decoder_type = config.decoder.type or "MLPDecoder"
        self.decoder = getattr(encoders, decoder_type)(config.decoder.args)

        # loss
        config.training = config.get('training')
        rec_config = config.training.get('reconstruction', OmegaConf.create())
        self.reconstruction_loss = get_distortion_loss(rec_config)
        reg_config = config.training.get('regularization', OmegaConf.create())
        self.regularization_loss = get_regularization_loss(reg_config)
        self.prior = getattr(priors, config.training.get('prior', "isotropic_gaussian"))

        # dimensionality reduction
        dim_red_type = config.get('dim_red', "PCA")
        self.dimred_type = getattr(decomposition, dim_red_type)
        self.dimred_batches = config.get('dim_red_batches', 128)
        self.register_buffer("latent_dimred", torch.eye(config.latent.dim))
        self.register_buffer("latent_mean", torch.zeros(config.latent.dim))
        self.register_buffer("fidelity", torch.zeros(config.latent.dim))
        self._latent_buffer = []

        # load from checkpoint
        if config_checkpoint:
            self.import_checkpoint(config_checkpoint)
        # record configs
        self.save_config(self.config)

    def get_encoder_shape(self, input_shape):
        if self.config.get('conditioning') is None:
            return input_shape
        concat_dims = []
        for t in self.config.conditioning.tasks:
            if "encoder" in t.get('target'):
                concat_dims.append(t['dim'])
        if len(input_shape) == 1:
            input_shape = (input_shape[0] + int(np.sum(concat_dims)), )
        else:
            input_shape = (input_shape[0] + int(np.sum(concat_dims)), *input_shape[1:])
        return input_shape
        
    def get_decoder_shape(self, latent_dim):
        if self.config.get('conditioning') is None:
            return latent_dim
        concat_dims = []
        for t in self.config.conditioning.tasks:
            if "decoder" in t.get('target'):
                concat_dims.append(t['dim'])
        if isinstance(latent_dim, (list, tuple)):
            if len(latent_dim) == 1:
                latent_dim = (latent_dim[0] + int(np.sum(concat_dims)), )
            else:
                latent_dim = (latent_dim[0] + int(np.sum(concat_dims)), *latent_dim[1:])
        else:
            latent_dim = latent_dim + int(np.sum(concat_dims))
        return latent_dim

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

    def get_beta(self):
        beta = self.config.training.beta if self.config.training.beta is not None else 1.0
        if self.config.training.warmup:
            beta_schedule_type = self.config.training.get('beta_schedule_type', "epoch")
            if beta_schedule_type == "epoch":
                beta = min(int(self.trainer.current_epoch) / self.config.training.warmup, beta)
            elif beta_schedule_type == "batch":
                beta = min(int(self.trainer.global_step) / self.config.training.warmup, beta)
        return beta

    # External methods
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # in lightning, forward defines the prediction/inference actions
        return self.encode(x, *args, **kwargs)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def sample(self, z_params: Union[dist.Distribution, torch.Tensor]) -> torch.Tensor:
        """Samples a latent distribution."""
        if isinstance(z_params, dist.Distribution):
            z = z_params.rsample()
        else:
            z = z_params
        return z

    def decode(self, z: torch.Tensor):
        """Decode an incoming tensor."""
        return self.decoder(z)

    def predict(self, z: torch.Tensor, y: Dict[str, torch.Tensor] = None):
        if not hasattr(self, "prediction_modules"):
            return y, None
        y_params = {}
        y_out = {} if y is None else y
        for task_name, predictor in self.prediction_modules.items():
            prediction_out = predictor(z)
            y_params[task_name] = prediction_out
            if y is not None:
                y_out[task_name] = prediction_out.sample()
        return y_out, y_params

    def reinforce(self, x, z, mode="forward"):
        """callback used for adversarial reinforcement"""
        if mode == "forward":
            return self.reconstruct(x)[1]
        elif mode == "latent":
            return self.decode(z)

    @torch.jit.export
    def full_forward(self, x: torch.Tensor, batch_idx: int=None,  sample: bool=True) -> Dict:
        """encodes and decodes an incoming tensor."""
        if isinstance(x, (tuple, list)):
            x, y = x
        else:
            x, y = x, None
        # encoding
        encoder_input = self.get_encoder_input(x, y=y)
        z_params = self.encoder(encoder_input)
        # sampling
        if sample:
            z = self.sample(z_params)
        else:
            z = z_params.mean
        # predicting
        y, y_params = self.predict(z=z, y=y)
        # decoding
        decoder_input = self.get_decoder_input(z, y=y)
        x_hat = self.decoder(decoder_input)
        return x_hat, z_params, z, y_params, y

    def get_encoder_input(self, x: torch.Tensor, y: Dict[str, torch.Tensor]=None):
        for k, v in self.concat_embeddings.items():
            if "encoder" in self.conditionings[k].target:
                assert y is not None, "model is conditioned; please explicit conditioning with the y keyword"
                assert y.get(k) is not None, "missing information for task %s"%k
                cond = v(y[k])
                if len(self.input_shape) == 1:
                    x = torch.cat([x, cond], 0)
                else:
                    extra_dims = torch.Size([1]*(len(self.input_shape) - 1))
                    cond = cond.view(cond.shape + extra_dims)
                    cond = cond.expand(*cond.shape[:-len(extra_dims)], *x.shape[-len(extra_dims):])
                    x = torch.cat([x, cond], dim=-(len(self.input_shape)))
        return x

    def get_decoder_input(self, z: torch.Tensor, y: Dict[str, torch.Tensor]=None):
        for k, v in self.concat_embeddings.items():
            if "decoder" in self.conditionings[k].target:
                assert y is not None, "model is conditioned; please explicit conditioning with the y keyword"
                assert y.get(k) is not None, "missing information for task %s"%k
                cond = v(y[k])
                if len(self.decoder.input_shape) == 1:
                    z = torch.cat([z, cond], -1)
                else:
                    extra_dims = torch.Size([1]*(len(self.input_shape) - 1))
                    cond = cond.view(cond.shape + extra_dims)
                    cond = cond.expand(*cond.shape[:-len(extra_dims)], *z.shape[-len(extra_dims):])
                    z = torch.cat([z, cond], dim=-(len(self.input_shape)))
        return z

    def trace(self, x: torch.Tensor, sample: bool = False):
        trace_model = {'encoder': {}, 'decoder' : {}}
        if isinstance(x, (tuple, list)):
            x, y = x
        x = self.get_encoder_input(x, y=y)
        z_params = self.encoder(x, trace = trace_model['encoder'])
        if sample:
            z = self.sample(z_params)
        else:
            z = z_params.mean
        y, y_params = self.predict(z, y)
        z = self.get_decoder_input(z, y=y)
        x_params = self.decoder(z, trace = trace_model['decoder'])
        trace = {}
        trace['embeddings'] = {'latent': {'data': z_params.mean, 'metadata': y}, **trace.get('embeddings', {})}
        trace['histograms'] = {**trace.get('histograms', {}),
                               **trace_distribution(z_params, name="latent", scatter_dim=True),
                               **trace_distribution(x_params, name="out"),
                               **trace_model}
        if y_params is not None:
            for t in y_params:
                trace['histograms'][f'pred_{t}'] = y_params[t].sample()
        return trace

    def loss(self, batch, x, z_params, z, y_params = None, drop_detail = False, **kwargs):
        if isinstance(batch, (tuple, list)):
            batch, y = batch
        else:
            batch, y = batch, None
        # reconstruction losses
        rec_loss = self.reconstruction_loss(x, batch, drop_detail=drop_detail)
        # regularization losses
        prior = self.prior(z.shape, device=batch.device)
        reg_loss = self.regularization_loss(prior, z_params, drop_detail=drop_detail)
        if drop_detail:
            rec_loss, rec_losses = rec_loss
            reg_loss, reg_losses = reg_loss
        # classic ELBO
        beta = self.get_beta()
        loss = rec_loss + beta * reg_loss
        # classification losses for semi-supervised learning
        classif_losses = {}
        if y_params is not None:
            for t, p in y_params.items():
                # if not p.get('predict', False):
                #     continue
                classif_losses[f"{t}_entropy"] = - p.entropy().mean(0).sum()
                if y is not None:
                    classif_losses[f"{t}_crossentropy"] =  - p.log_prob(y[t]).mean(0).sum()
                    loss = loss + self.conditionings[t].get('weight', 1.0) * classif_losses[f"{t}_crossentropy"]
                else:
                    loss = loss + self.conditionings[t].get('weight', 1.0) * classif_losses[f"{t}_entropy"]
        if drop_detail:
            return loss, {"full_loss": loss.cpu().detach(), **classif_losses, **reg_losses, **rec_losses}
        else:
            return loss

    def _update_latent_buffer(self, z_params):
        if isinstance(z_params, dist.Normal):
            z_params = z_params.mean
        if len(self._latent_buffer) <= self.dimred_batches:
            self._latent_buffer.append(z_params.detach().cpu())

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, z_params, z, y_params, y = self.full_forward(batch, batch_idx)
        loss, losses = self.loss(batch, x, z_params, z, y_params=y_params, epoch=self.trainer.current_epoch, drop_detail=True)
        losses['beta'] = self.get_beta()
        self.log_losses(losses, "train", prog_bar=True)
        self._update_latent_buffer(z_params)
        return loss

    def on_train_epoch_end(self):
        # shamelessly taken from Caillon's RAVE
        latent_pos = torch.cat(self._latent_buffer, 0).reshape(-1, self.config.latent.dim)
        self.latent_mean.copy_(latent_pos.mean(0))

        dimred = self.dimred_type(n_components=latent_pos.size(-1))
        dimred.fit(latent_pos.cpu().numpy())
        components = dimred.components_
        components = torch.from_numpy(components).to(latent_pos)
        self.latent_dimred.copy_(components)

        var = dimred.explained_variance_ / np.sum(dimred.explained_variance_)
        var = np.cumsum(var)
        self.fidelity.copy_(torch.from_numpy(var).to(self.fidelity))

        var_percent = [.8, .9, .95, .99]
        for p in var_percent:
            self.log(f"{p}%_manifold",
                    np.argmax(var > p).astype(np.float32))
        self.trainer.logger.experiment.add_image("latent_dimred", components.unsqueeze(0).numpy(), self.trainer.current_epoch)
        self._latent_buffer = []
        
    def validation_step(self, batch, batch_idx):
        x, z_params, z, y_params, y = self.full_forward(batch, batch_idx)
        loss, losses = self.loss(batch, x, z_params, z, y_params=y_params, drop_detail=True)
        self.log_losses(losses, "valid", prog_bar=True)
        return loss

    def reconstruct(self, x, *args, sample_latent=False, sample_data=False, **kwargs):
        x_out, _, _, _, _ = self.full_forward(_recursive_to(x, self.device), sample=sample_latent)
        if isinstance(x_out, dist.Distribution):
            if sample_data:
                x_out = x_out.sample()
            else:
                x_out = x_out.mean
        return x, x_out

    def sample_from_prior(self, n_samples=1, temperature=1.0, sample=False):
        temperature = checklist(temperature)
        generations = []
        y = {}
        if self.conditionings is not None:
            decoder_task = list(filter(lambda x: "decoder" in self.conditionings[x]['target'], self.concat_embeddings.keys()))
            if len(decoder_task) > 0:
                for t in decoder_task:
                    y[t] = torch.randint(0, self.conditionings[t]['dim'], (n_samples,), device=self.device)
        for t in temperature:
            z = torch.randn((n_samples, self.latent.dim), device=self.device) * t
            decoder_input = self.get_decoder_input(z, y)
            x = self.decode(decoder_input)
            if sample:
                x = x.sample()
            else:
                x = x.mean
            generations.append(x)
        return torch.stack(generations, 1)

    def get_scripted(self, mode: str = "audio", script: bool=True, **kwargs):
        if mode == "audio":
            scriptable_model = ScriptableAudioAutoEncoder(self, **kwargs)
        else:
            raise ValueError("error while scripting model %s : mode %s not found"%(type(self), mode))
        if script:
            return torch.jit.script(scriptable_model)
        else:
            return scriptable_model



class InfoGAN(GAN):
    def __init__(self, config=None, encoder=None, decoder=None, discriminator=None, training=None, latent=None, **kwargs) -> None:
        super(GAN, self).__init__()
        if isinstance(config, dict):
            config = OmegaConf(config)
        else:
            config = OmegaConf.create()

        input_shape = config.get('input_shape') or kwargs.get('input_shape')
        # setup latent
        config.latent = config.get('latent') or latent or {}
        self.prior = getattr(priors, config.latent.get('prior', "isotropic_gaussian"))
       # latent configs
        config.latent = config.get('latent') or latent
        self.latent = config.latent
        # encoder architecture
        config.encoder = config.get('encoder') or encoder
        config.encoder.args = config.encoder.get('args', {})
        if config.encoder['args'].get('input_shape') is None:
            config.encoder['args']['input_shape'] = config.get('input_shape') or kwargs.get('input_shape')
        if config.encoder['args'].get('target_shape') is None:
            config.encoder['args']['target_shape'] = config.latent.dim
        config.encoder['args']['target_dist'] = config.latent.dist
        encoder_type = config.encoder.type or "MLPEncoder"
        self.encoder = getattr(encoders, encoder_type)(config.encoder.args)
        # decoder architecture
        config.decoder = config.get('decoder', decoder)
        config.decoder.args = config.decoder.get('args', {})
        config.decoder.args.input_shape = config.latent.dim
        if config.decoder.args.get('input_shape') is None:
            config.decoder.args.input_shape = config.latent.dim
        if config.decoder.args.get('target_shape') is None:
            config.decoder.args.target_shape = config.get('input_shape') or kwargs.get('input_shape')
        decoder_type = config.decoder.type or "MLPDecoder"
        self.decoder = getattr(encoders, decoder_type)(config.decoder.args) 
        # setup discriminator
        config.discriminator = config.get('discriminator') or discriminator 
        config.discriminator.args.input_shape = input_shape
        self.init_discriminator(config.discriminator)
        # setup training
        config.training = config.get('training') or training
        config.training.mode = config.training.get('mode', 'adv')
        assert config.training.mode in self.gan_modes
        self.automatic_optimization = False
        self.reconstruction_losses = parse_additional_losses(config.training.get('rec_losses'))
        reg_config = config.training.get('regularization_loss', OmegaConf.create())
        self.regularization_loss = getattr(regularization, reg_config.get('type', "KLD"))(**reg_config.get('args',{}))
        self.prior = getattr(priors, config.training.get('prior', "isotropic_gaussian"))

        self.config = config
        self.save_hyperparameters(dict(self.config))
        self.__dict__['generator'] = self.decoder

    def get_parameters(self, parameters=None, model=None, prefix=None):
        model = model or self
        if parameters is None:
            params = list(model.parameters())
        else:
            full_params = dict(model.named_parameters())
            full_params_names = list(full_params.keys())
            full_params_names_prefix = full_params_names if prefix is None else [f"{prefix}.{n}" for n in full_params_names]
            params = []
            for param_regex in parameters:
                valid_names = list(filter(lambda x: re.match(param_regex, full_params_names_prefix[x]), range(len(full_params_names))))
                params.extend([full_params[full_params_names[i]] for i in valid_names])
        return params


    def configure_optimizers(self):
        gen_p = self.get_parameters(self.config.get('optim_params'), self.encoder, "encoder")
        gen_p += self.get_parameters(self.config.get('optim_params'), self.decoder, "decoder")
        dis_p = self.get_parameters(self.config.get('optim_params'), self.discriminator, "discriminator")
        if len(gen_p) == 0:
            dis_opt = torch.optim.Adam(dis_p, 1e-4, (.5, .9))
            return dis_opt
        elif len(dis_p) == 0:
            gen_opt = torch.optim.Adam(gen_p, 1e-4, (.5, .9))
            return gen_opt
        else:
            dis_opt = torch.optim.Adam(dis_p, 1e-4, (.5, .9))
            gen_opt = torch.optim.Adam(gen_p, 1e-4, (.5, .9))
            return gen_opt, dis_opt    

    def encode(self, x):
        return self.encoder(x)

    def sample_prior(self, batch=None, shape=None):
        if batch is None:
            return super().sample_prior(batch=batch, shape=shape)
        else:
            z = self.encode(batch)
            return z

    def generate(self, x=None, z=None, sample=True, **kwargs):
        if isinstance(z, dist.Distribution):
            z = z.rsample()
        out = self.decoder(z, **kwargs)
        return out

    def generator_loss(self, generator, batch, out, d_fake, z_params, hidden=None, **kwargs):
        adv_loss = super().generator_loss(generator, batch, out, d_fake, z_params, hidden=hidden)
        z = z_params.sample()
        prior = self.prior(z.shape, device=batch.device)
        reg_loss = self.regularization_loss(prior, z_params)
        beta = self.config.training.beta if self.config.training.beta is not None else 1.0
        if self.config.training.warmup:
            beta = min(int(self.trainer.current_epoch) / self.config.training.warmup, beta)
        return adv_loss + beta * reg_loss
