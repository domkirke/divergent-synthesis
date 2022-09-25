import os
from collections import OrderedDict
import torch, torchvision as tv
import sklearn
import numpy as np
from omegaconf import OmegaConf
from typing import Optional, OrderedDict, Sequence
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage

from divergent_synthesis import models
from divergent_synthesis.utils import checkdir
import csv

class ImprovedPR():
    
    def __init__(self,
                 neighborhood: Sequence[int] = [3], 
                 eps: float = 1e-5):
                 
        """
        Initialize the Improved Precision & Recall metric class

        Parameters
        ----------
        device : str
            The computing device.
        model : str
            The (optional) external model which produces the embeddings.
        neighborhood : list of int
            Number of neighbors used to estimate the manifold.
        eps: float
            Small number for numerical stability.
        n_samples : int
            The number of samples to compute the metric on.
            If it's set to None it will use all the available samples.
        pre_compute : bool
            Whether to pre-compute embeddings or not.
            If set to False embeddings will be computed for all samples at once.

        """
        self.pre_compute=True
        self.neighborhood = neighborhood
        self.eps = eps

    def __call__(self, 
                 x_r: torch.Tensor, 
                 x_g: torch.Tensor) -> torch.Tensor:
        """
        Calculates k-NN precision and recall for two sets of feature vectors.
    
        Args:
            ref_features (np.array/tf.torch.Tensor): Feature vectors of reference images.
            eval_features (np.array/tf.torch.Tensor): Feature vectors of generated images.
            nhood_sizes (list): Number of neighbors used to estimate the manifold.
            row_batch_size (int): Row batch size to compute pairwise distances
                (parameter to trade-off between memory usage and performance).
            col_batch_size (int): Column batch size to compute pairwise distances.
            num_gpus (int): Number of GPUs used to evaluate precision and recall.
        Returns:
            State (dict): Dict that contains precision and recall calculated from
            ref_features and eval_features.
        """
        # Embed both distributions
        phi_g = x_g
        phi_r = x_r
        if (not self.pre_compute):
            phi_g = self._embed(x_g)
            phi_r = self._embed(x_r)
        # TODO: Check if we can avoid going back to CPU
        phi_g = phi_g.cpu()
        phi_r = phi_r.cpu()
        r_manifold = self._compute_manifold(phi_r)
        g_manifold = self._compute_manifold(phi_g)
        # Precision: How many points from generated features are in real manifold.
        precision = self._evaluate_manifold(r_manifold, phi_r, phi_g)
        precision = precision.mean(axis=0)
        # Recall: How many points from real features are in generative manifold.
        recall = self._evaluate_manifold(g_manifold, phi_g, phi_r)
        recall = recall.mean(axis=0)
        return precision, recall
    
    def _compute_manifold(self, 
                          features, 
                          clamp_to_percentile=None):
        """Estimate the manifold of given feature vectors.
        
            Args:
                distance_block: DistanceBlock object that distributes pairwise distance
                    calculation to multiple GPUs.
                features (np.array/torch.Tensor): Matrix of feature vectors to estimate their manifold.
                row_batch_size (int): Row batch size to compute pairwise distances
                    (parameter to trade-off between memory usage and performance).
                col_batch_size (int): Column batch size to compute pairwise distances.
                nhood_sizes (list): Number of neighbors used to estimate the manifold.
                clamp_to_percentile (float): Prune hyperspheres that have radius larger than
                    the given percentile.
                eps (float): Small number for numerical stability.
        """
        # Estimate manifold of features by calculating distances to k-NN of each sample.
        seq = np.arange(max(self.neighborhood) + 1, dtype=np.int32)
        # Compute pairwise distances
        distances = self._compute_pairwise_distance(features)
        # Find the k-nearest neighbor from the current batch.
        manifold = np.partition(distances, seq, axis=1)[:, self.neighborhood]
        if clamp_to_percentile is not None:
            max_distances = np.percentile(manifold, clamp_to_percentile, axis=0)
            manifold[manifold > max_distances] = 0
        return manifold
    
    def _evaluate_manifold(self, 
                           manifold, 
                           ref_features, 
                           eval_features, 
                           return_realism=False, 
                           return_neighbors=False):
        """Evaluate if new feature vectors are at the manifold."""
        # Compute pairwise distances
        distances = self._compute_pairwise_distance(eval_features, ref_features)
        # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
        # If a feature vector is inside a hypersphere of some reference sample, then
        # the new sample lies at the estimated manifold.
        # The radii of the hyperspheres are determined from distances of neighborhood size k.
        samples_in_manifold = distances <= manifold
        batch_predictions = np.any(samples_in_manifold, axis=1).astype(np.int32)
        max_realism_score = np.max(manifold[:, 0] / (distances + self.eps), axis=1)
        nearest_indices = np.argmin(distances, axis=1)
        if return_realism and return_neighbors:
            return batch_predictions, max_realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, max_realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices
        return batch_predictions
            
    def _compute_pairwise_distance(self, 
                                   data_x, 
                                   data_y=None):
        """
        Args:
            data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
            data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
        Returns:
            numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
        """
        if data_y is None:
            data_y = data_x
        dists = sklearn.metrics.pairwise_distances(data_x, data_y, metric='euclidean', n_jobs=8)
        return dists


class CSVWriter(object):
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        with open(filename, 'w+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row_dict):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(dict(row_dict))


class EvaluationCallback(Callback):
    def __init__(self, model_path=None, feature_path=None, monitor_epochs=5, n_batches=16, batch_size=1024, label=None, logdir=None):
        assert model_path is not None
        self.original_model, self.model_config = self.get_model_from_path(model_path) 
        self.feature_model = None; self.feature_config = None
        self.label = label or ""
        if feature_path is not None:
            self.feature_model, self.feature_config= self.get_feature_model(feature_path)
            self.label = os.path.basename(os.path.dirname(feature_path))
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.monitor_epochs = monitor_epochs
        self.temperatures = [0.1, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 3.0] 
        self.logdir = logdir
        if self.logdir is not None:
            self.logdir = os.path.join(self.logdir, "evaluations")
            checkdir(self.logdir)
            self.precision_writer = CSVWriter(os.path.join(self.logdir, "precision.csv"), ["t=%s"%t for t in self.temperatures])
            self.recall_writer = CSVWriter(os.path.join(self.logdir, "recall.csv"), ["t=%s"%t for t in self.temperatures])
            self.variance_writer = CSVWriter(os.path.join(self.logdir, "variance.csv"), ["t=%s"%t for t in self.temperatures])

        self.pr = ImprovedPR()
    
    def get_model_from_path(self, model_path: str):
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
        if isinstance(model, models.Classifier):
            model.classifiers['class'].dist_module = None
            return model.classifiers['class'].eval(), config
        else:
            return model.eval(), config

    def get_feature_model(self, model_id: str):
        if os.path.isdir(model_id):
            model, config = self.get_model_from_path(model_id)
            input_shape = config.model.input_shape
            self.normalize = False
        else:
            model = getattr(tv.models, model_id)(pretrained=True)
            input_shape = (3, 299, 299) if model_id == "inception_v3" else (3, 256, 256)
            self.normalize=True
        return model.eval(), input_shape

    def on_validation_epoch_end(self, trainer: pl.Trainer=None, pl_module: pl.LightningModule=None):
        if trainer is not None:
            with torch.no_grad():
                if trainer.state.stage == RunningStage.SANITY_CHECKING:
                    return
                if trainer.current_epoch % self.monitor_epochs != 0:
                    return
        # make batch loader
        device = next(pl_module.parameters()).device
        self.original_model.eval()
        pl_module.eval()
        if device != next(self.original_model.parameters()).device:
            self.original_model = self.original_model.to(device)
            if self.feature_model is not None:
                self.feature_model = self.feature_model.to(device)
        with torch.no_grad():
            for i in range(self.n_batches):
                batch_shape = (self.batch_size, len(self.temperatures))
                original_samples = self.original_model.sample_from_prior(self.batch_size, self.temperatures)
                deviated_samples = pl_module.sample_from_prior(self.batch_size, self.temperatures)
                if self.feature_model is not None:
                    original_samples = self.feature_model(original_samples.view(-1, *original_samples.shape[2:]))
                    deviated_samples = self.feature_model(deviated_samples.view(-1, *deviated_samples.shape[2:]))
                    original_samples = original_samples.view(*batch_shape, *original_samples.shape[1:])
                    deviated_samples = deviated_samples.view(*batch_shape, *deviated_samples.shape[1:])
                # original_samples = original_samples.cpu().numpy()
                # deviated_samples = deviated_samples.cpu().numpy()
            precision = OrderedDict({}); recall = OrderedDict({}); variance = OrderedDict({})
            for i, t in enumerate(self.temperatures):
                precision["t=%.1f"%t], recall["t=%.1f"%t] = self.pr(original_samples[:, i], deviated_samples[:, i])
                variance["t=%.1f"%t] = deviated_samples.mean().detach().cpu()
            if trainer is not None:
                for k, v in precision.items():
                    trainer.logger.experiment.add_scalar('precision_%s/t=%s'%(self.label, k), precision[k], trainer.current_epoch)
                    trainer.logger.experiment.add_scalar('recall_%s/t=%s'%(self.label, k), recall[k], trainer.current_epoch)
                    trainer.logger.experiment.add_scalar('variance_%s/t=%s'%(self.label, k), recall[k], trainer.current_epoch)
        if self.logdir is not None:
            self.precision_writer.log(precision)
            self.recall_writer.log(recall)
            self.variance_writer.log(variance)
        return {'precision': precision, 'recall': recall, 'variance': variance}
            
            