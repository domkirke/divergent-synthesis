import argparse
import torch, GPUtil as gpu, argparse, os, tqdm, torchvision as tv
from torch import distributions as dist
from divergent_synthesis import data, models 
from divergent_synthesis.utils.misc import _recursive_to
from divergent_synthesis.utils import get_model_from_path, checkdir
from divergent_synthesis.losses import LogDensity, MSE, KLD, MMD
from torchmetrics import ConfusionMatrix
from divergent_synthesis.monitor.callbacks import EvaluationCallback 

# detect CUDA devices
accelerator = "cpu"
device = torch.device('cpu')
if torch.cuda.is_available():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices != "":
        device = torch.device('cuda:'+visible_devices.split(",")[0])
    elif gpu.getAvailable(maxMemory=0.05):
        available_devices = gpu.getAvailable(maxMemory=0.05)
        if len(available_devices) > 0:
            device = torch.device('cuda:'+str(available_devices[0]))
elif hasattr(torch.backends, "mps"):
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = torch.device('mps')

def accum_losses(full_losses, loss):
    for k, v in loss.items():
        if k in full_losses:
            full_losses[k].append(_recursive_to(v, torch.device('cpu')))
        else:
            full_losses[k] = [_recursive_to(v, torch.device('cpu'))]

def get_losses(model, train_loader, args):
    with torch.no_grad():
        full_losses = {}
        current_batch = 0
        confusion_matrix = ConfusionMatrix(10)
        # classic losses
        for x, y in tqdm.tqdm(iter(train_loader), desc="obtaining losses...", total=len(train_loader)):
            x = x.to(device)
            y = _recursive_to(y, device)
            x_out, z_params, z, = None, None, None
            if type(model) == models.AutoEncoder:
                x_out, z_params, z, _, _ = model.full_forward(x)
            elif type(model) == models.Classifier:
                y_out = model.classify(x, y)['class']
            if "ld" in args.losses:
                loss, losses = LogDensity()(x_out, x, drop_detail=True)
                accum_losses(full_losses, losses) 
            if "mse" in args.losses:
                loss, losses = MSE()(x_out, x, drop_detail=True)
                accum_losses(full_losses, losses) 
            if "kld" in args.losses:
                assert isinstance(model, models.AutoEncoder), "kld only with AutoEncoder"
                loss, losses = KLD()(z_params, model.prior(z.shape, device=z.device), drop_detail=True)
                accum_losses(full_losses, losses)
            if "classif" in args.losses:
                assert isinstance(model, models.Classifier), "classif only with Classifier"
                # get confusion matrix
                _, log_y = LogDensity()(y_out, y['class'], drop_detail=True)
                if isinstance(y_out, dist.Categorical):
                    y_out = y_out.sample()
                confusion_matrix.update(y_out.cpu(), y['class'].cpu())
                accum_losses(full_losses, log_y)
            current_batch += 1
            if args.batches_max is not None:
                if current_batch > args.batches_max:
                    break
        # classification score
        if "classif" in args.losses:
            cmat = confusion_matrix.compute()
            full_losses = {**full_losses, 'classif_score': (cmat.diag().sum() / cmat.sum())*100}
        # precision / recall
        if "pr" in args.losses:
            eval_args = {'model_path':args.original_model,
                     'feature_path': args.feature_model,
                     'batch_size':args.batch_size,
                     'max_batches':args.n_batches}
            pr_monitor = EvaluationCallback(**eval_args)
            precision, recall = pr_monitor.on_validation_end()
            full_losses = {**precision, **recall, **full_losses}
        return full_losses


def get_generations(model, args, out_path):
    generations = None
    if args.generate > 0:
        assert isinstance(model, (models.AutoEncoder, models.DivergentGenerative))
        out = model.sample_from_prior(n_samples = args.generate, temperature=args.temperatures)
        out = out.reshape(out.size(0) * out.size(1), *out.shape[2:])
        generations = tv.utils.make_grid(out, nrow=len(args.temperatures), value_range=[0, 1]) 
        checkdir(out_path)
        filename = os.path.join(out_path, "generations_"+args.model.split('/')[-2]+".jpg")
        tv.utils.save_image(generations, filename)
        

if __name__ == "__main__":
    valid_losses = ['pr', 'ld', 'mse', 'kld', 'classif']
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default="", help="path to model")
    parser.add_argument('-l', '--losses', type=str, default=['ld', 'mse', 'kld'], choices=valid_losses, nargs="*", help="losses to compute")
    parser.add_argument('-g', '--generate', type=int, default=0, help="number of generated examples (default: 0)")
    parser.add_argument('-o', '--output', default="generations/", type=str)
    parser.add_argument('--classif_model', type=str, default=None, help="id for generative losses embeddings")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size (default: 256)")
    parser.add_argument('--batches_max', default=None, type=int, help="maximum number of batches (default: none)")
    parser.add_argument('--temperatures', default=[0.2, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0, 10.0], type=float, help="maximum number of batches (default: none)")


    args = parser.parse_args()

    # load model
    model, config = get_model_from_path(args.model)
    model = model.to(device)
    # load data
    data_module = getattr(data, config.data.module)(config.data)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    # get losses
    print('-- training set')
    losses_train = get_losses(model, data_module.train_dataloader(batch_size=args.batch_size), args)
    print('-- validation set')
    losses_valid = get_losses(model, data_module.val_dataloader(batch_size=args.batch_size), args)
    final_losses = {'train': {}, 'valid': {}}
    for k in losses_train.keys():
        final_losses['train'][k] = torch.Tensor(losses_train[k]).mean()
        final_losses['valid'][k] = torch.Tensor(losses_valid[k]).mean()
    print(final_losses)
    generations = get_generations(model, args, args.output)


