import argparse, pdb, os, sys
import logging
from omegaconf import OmegaConf, DictConfig
import torch, pytorch_lightning as pl, hydra
from pytorch_lightning.loggers import TensorBoardLogger
import GPUtil as gpu
from divergent_synthesis import data, models, get_callbacks
from divergent_synthesis.utils import save_trainig_config, get_root_dir
logger = logging.getLogger(__name__)
torch.multiprocessing.set_sharing_strategy('file_system')


# detect CUDA devices
accelerator = "cpu"
devices = None
if torch.cuda.is_available():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices != "":
        accelerator = "gpu"
        devices = visible_devices.split(",")[0]
    elif gpu.getAvailable(maxMemory=0.05):
        available_devices = gpu.getAvailable(maxMemory=0.05)
        if len(available_devices) > 0:
            accelerator = "gpu"
            devices = [available_devices[0]]
elif hasattr(torch.backends, "mps"):
    if torch.backends.mps.is_available():
        accelerator = "mps"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# main routine
@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)
    # set num workers
    config.data.loader['num_workers'] = config.data.loader.get('num_workers', os.cpu_count())
    # get data
    if config.get('data'):
        data_module = getattr(data, config.data.module)(config.data)
    # get model
    config.model.input_shape = data_module.shape
    model = getattr(models, config.model.type)(config.model)
    # load checkpoint in case
    if config.get('checkpoint'):
        model = model.load_from_checkpoint(config.checkpoint)
    # import callbacks
    # setup trainer
    trainer_config = config.get('pl_trainer', {})
    trainer_config['accelerator'] = accelerator
    if devices:
        trainer_config['devices'] = devices
    trainer_config['default_root_dir'] = get_root_dir(config.rundir, config.name)
    logger = TensorBoardLogger(save_dir=os.path.join(trainer_config['default_root_dir']), version="")
    callbacks = get_callbacks(config.get('callbacks'), trainer_config['default_root_dir'])
    trainer = pl.Trainer(**trainer_config, callbacks=callbacks)
    # optional breakpoint before training
    if bool(config.get('check')):
        pdb.set_trace()
    # train!
    save_trainig_config(config, data_module, path=trainer_config['default_root_dir'])
    if config.get('seed') is not None:
        torch.manual_seed(int(config.seed))
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()