from pytorch_lightning import callbacks   
import sys; sys.path.append('../')
import pdb
from divergent_synthesis.utils import checkdir
from pytorch_lightning.trainer.states import RunningStage

class ModelCheckpoint(callbacks.ModelCheckpoint):
    def __init__(self, dirpath=None, filename=None, epoch_schedule=None, epoch_period=None, **kwargs):
        dirpath += f"/{filename}"
        super(ModelCheckpoint, self).__init__(dirpath=dirpath, filename=filename, **kwargs)
        self.epoch_period = epoch_period
        self.epoch_schedule = epoch_schedule
        if self.epoch_period is not None:
            checkdir(self.dirpath+"/epochs")

    def on_epoch_end(self, trainer, pl_module):
        if trainer.state.stage == RunningStage.SANITY_CHECKING:
            return
        if self.epoch_period is not None:
            if trainer.current_epoch % self.epoch_period == 0:
                filepath = f"{self.dirpath}/epochs/{self.filename}_{self.STARTING_VERSION}_{trainer.current_epoch}{self.FILE_EXTENSION}"
                trainer.save_checkpoint(filepath, self.save_weights_only)
        if self.epoch_schedule is not None:
            if trainer.current_epoch in self.epoch_schedule:
                filepath = f"{self.dirpath}/epochs/{self.filename}_{self.STARTING_VERSION}_{trainer.current_epoch}{self.FILE_EXTENSION}"
                trainer.save_checkpoint(filepath, self.save_weights_only)



