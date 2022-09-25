import numpy as np, matplotlib.pyplot as plt
import torch, torch.distributions as dist, sys, pdb
sys.path.append("../")
import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from torchmetrics import ConfusionMatrix

def _recursive_to(obj, device):
    if isinstance(obj, dict):
        return {k: _recursive_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_recursive_to(o, device) for o in obj]
    elif torch.is_tensor(obj):
        return obj.to(device=device)
    else:
        raise TypeError('type %s not handled by _recursive_to'%type(obj))

def render_confusion_matrix(ax, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    ax.tick_params(axis='x', labelsize=10)
    plt.yticks(tick_marks, classes)
    ax.tick_params(axis='y', labelsize=10)

    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class ClassificationMonitor(Callback):

    def __init__(self, n_batches=5, batch_size=1024, monitor_epochs=1):
        """
        Callback for monitoring and dissecting the model's guts.
        Args:
            n_batches (int): number of batches (default: 5)
            batch_size (int): batch size (default: 1024)
            monitor_epochs (int): monitoring period in epochs (default: 1)
            embedding_epochs (int): embedding plot period in epochs (default: 10)
        """
        super().__init__()
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.monitor_epochs = monitor_epochs

    def get_confusion_matrices(self, pl_module, loader):
        confusion_matrices = {}
        device = next(pl_module.parameters()).device
        for task, task_config in pl_module.config.classifiers.items():
            confusion_matrices[task] = ConfusionMatrix(num_classes=task_config['dim']).to(device)
        for i, data in enumerate(loader):
            data = _recursive_to(data, device)
            x, y = data 
            out = pl_module.classify(x, y)
            for task, prediction in out.items():
                if isinstance(prediction, dist.Categorical):
                    prediction = prediction.sample()
                confusion_matrices[task].update(prediction, y[task])
        for task, cm in confusion_matrices.items():
            confusion_matrices[task] = cm.compute().cpu()
        return confusion_matrices

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        with torch.no_grad():
            if trainer.state.stage == RunningStage.SANITY_CHECKING:
                return
            if trainer.current_epoch % self.monitor_epochs != 0:
                return
            # plot model parameters distribution
            if hasattr(trainer.datamodule, "train_dataloader"):
                loader = trainer.datamodule.train_dataloader(batch_size=self.batch_size, drop_last=False)
                confusion_matrices = self.get_confusion_matrices(pl_module, loader)
                for task, cm in confusion_matrices.items():
                    classes = range(pl_module.config.classifiers[task].dim)
                    fig, ax = plt.subplots(1, 1)
                    render_confusion_matrix(ax, cm, classes)
                    trainer.logger.experiment.add_figure('confusion_%s/train'%task, fig, trainer.current_epoch)
            if hasattr(trainer.datamodule, "validation_loader"):
                loader = trainer.datamodule.valid_dataloader(batch_size=self.batch_size, drop_last=False)
                confusion_matrices = self.get_confusion_matrices(pl_module, loader)
                for task, cm in confusion_matrices.items():
                    classes = range(pl_module.config.classifiers[task].dim)
                    fig, ax = plt.subplots(1, 1)
                    cm_fig = render_confusion_matrix(ax, cm, classes)
                    trainer.logger.experiment.add_figure('confusion_%s/valid'%task, cm_fig, trainer.current_epoch)
