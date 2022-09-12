from typing import Iterable
import torch, torch.distributions as dist, torchvision as tv, pdb, random, matplotlib.pyplot as plt, os, numpy as np, re
import torchaudio, tqdm, dill
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from divergent_synthesis.utils import checkdir

def check_mono(sgl, normalize=True):
    sgl = sgl.squeeze()
    if sgl.ndim == 2:
        if sgl.shape[0] > 1:
            sgl = sgl.mean(0).unsqueeze(0)
    if normalize:
        sgl /= sgl.max()
    return sgl

def fit_data(data, target_shape, has_batch = True):
    idx_range = range(int(has_batch), len(target_shape))
    for i in idx_range:
        if data.shape[i] > target_shape[i]:
            idx = (slice(None),)*i + (slice(0, target_shape[i]),)
            data = data.__getitem__(idx)
        elif data.shape[i] < target_shape[i]:
            pdb.set_trace()
            pad_shape = tuple(target_shape[:i]) + (target_shape[i] - data.shape[i],) + tuple(target_shape[i+1:])
            data = torch.cat([data, torch.zeros(pad_shape).to(data.device)])
    return data

class ImgReconstructionMonitor(Callback):

    def __init__(self, n_reconstructions: int = 5, n_morphings: int = 2, n_samples = 5,
                 temperature_range=None, monitor_epochs=1):
        """
        Callback for image reconstruction monitoring.
        Args:
            n_reconstructions (int): number of reconstructed examples (default: 5)
            n_morphings (int): number of latent translations (default: 2)
            n_samples (int): number of samples per temperature value for sample prior (default: 5)
            temperature_range: temperature for prior sampling (default : [0.1, 1.0, 3.0, 5.0, 10.0])
            monitor_epochs (int): monitoring period in epochs (default: 1)
            reconstruction_epochs: rec
        """
        self.n_reconstructions = n_reconstructions
        self.temperature_range = temperature_range or [0.1, 1.0, 3.0, 5.0, 10.0]
        self.n_samples = n_samples
        self.monitor_epochs = monitor_epochs

    def plot_reconstructions(self, model, loader):
        data = next(loader(batch_size=self.n_reconstructions).__iter__())
        x_original, x_out = model.reconstruct(data)
        value_range = [x_original.min(), x_original.max()]
        x_out = [x_tmp.cpu() for x_tmp in x_out]
        out = torch.stack([x_original, *x_out], 0).reshape((len(x_out) + 1) * x_original.shape[0], *x_original.shape[1:])
        return tv.utils.make_grid(out, nrow=x_original.shape[0], value_range=value_range)

    def plot_samples(self, model, datamodule):
        out = model.sample_from_prior(n_samples=self.n_samples, temperature=self.temperature_range)
        if isinstance(out, dist.Distribution):
            out = out.mean
        #out = out.transpose(0, 1)
        out = out.reshape(out.size(0) * out.size(1), *out.shape[2:])
        if hasattr(datamodule.transforms, "invert"):
            out = datamodule.transforms.invert(out)
        full_img = tv.utils.make_grid(out, nrow=self.n_samples, value_range=[0, 1]) 
        return full_img

    def reconstruct_file(self, model, files, dataset):
        originals = []; generations = []
        for f in files:
            data, meta = dataset.transform_file(f"{dataset.root_directory}/{f}")
            original, generation = model.reconstruct(data)
            originals.append(dataset.invert_transform(original))
            generations.append(dataset.invert_transform(generation))
        originals, generations = torch.cat(originals, -1).squeeze(), torch.cat(generations, -1).squeeze()
        return check_mono(originals, normalize=True), check_mono(generations, normalize=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        with torch.no_grad():
            if trainer.state.stage == RunningStage.SANITY_CHECKING:
                return
            if trainer.current_epoch % self.monitor_epochs != 0:
                return
            model = trainer.model
            # plot reconstructions
            if hasattr(model, 'reconstruct'):
                train_rec = self.plot_reconstructions(model, trainer.datamodule.train_dataloader)
                trainer.logger.experiment.add_image('reconstructions/train', train_rec, trainer.current_epoch)
                valid_rec = self.plot_reconstructions(model, trainer.datamodule.val_dataloader)
                trainer.logger.experiment.add_image('reconstructions/valid', valid_rec, trainer.current_epoch)
                if trainer.datamodule.test_dataset:
                    test_rec = self.plot_reconstructions(model, trainer.datamodule.test_dataloader)
                    trainer.logger.experiment.add_image('reconstructions/test', test_rec, trainer.current_epoch)

            # plot prior sampling
            if hasattr(model, 'sample_from_prior'):
                samples = self.plot_samples(model, trainer.datamodule)
                trainer.logger.experiment.add_image('samples', samples, trainer.current_epoch)

