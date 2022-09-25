import torch, os, torchvision as tv, torchmetrics
from omegaconf import OmegaConf
from divergent_synthesis.models.gans import GAN
from divergent_synthesis.models.classifier import Classifier
from .loss import Loss
from .. import models
from .regularization import MMD
from torch.autograd import Function
import numpy as np
import scipy.linalg


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
    model = model.load_from_checkpoint(model_path, config=config.model)
    if isinstance(model, Classifier):
        model.classifiers['class'].dist_module = None
        return model.classifiers['class'], config
    elif isinstance(model, GAN):
        return model, config


def sqrtm(matrix, tol = 1.e-6):
    """Compute the square root of a positive definite matrix."""
    # perform the decomposition
    s, v = torch.linalg.eigh(matrix, UPLO='U')
    # _, s, v = matrix.svd()  # passes torch.autograd.gradcheck()
    # truncate negative components
    above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
    s = s[..., above_cutoff]
    v = v[..., above_cutoff]
    # check if some eigenvalues are not too close (unstable gradients)
    # tooclose = torch.where((s[1:] - s[:-1]).abs() > tol)[0]
    # s = torch.cat([s[..., [0]], s[..., tooclose]], -1)
    # v = torch.cat([v[..., [0]], v[..., tooclose]], -1)
    # compose the square root matrix
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


# class MatrixSquareRoot(Function):
#     """
#     Shamelessy taken from https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py
#     Square root of a positive definite matrix.
#     NOTE: matrix square root is not differentiable for matrices with
#           zero eigenvalues.
#     """
#     @staticmethod
#     def forward(ctx, input):
#         m = input.detach().cpu().numpy().astype(np.float_)
#         sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
#         ctx.save_for_backward(sqrtm)
#         return sqrtm

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = None
#         if ctx.needs_input_grad[0]:
#             sqrtm, = ctx.saved_tensors
#             sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
#             gm = grad_output.data.cpu().numpy().astype(np.float_)

#             # Given a positive semi-definite matrix X,
#             # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
#             # matrix square root dX^{1/2} by solving the Sylvester equation:
#             # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
#             grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

#             grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
#         return grad_input
# sqrtm = MatrixSquareRoot.apply


eps = torch.finfo(torch.get_default_dtype()).eps


class BAD(Loss):
    def __repr__(self):
        return "InceptionScore(feature=%s, splits=%s)"%(self.feature, self.splits)
    
    def __init__(self, model='inception_v3', splits=10, reduction=None, polarity="unipolar", 
                 couple="mmd", mode="sum", weights=None, mmd_scale=1.0, tasks=None, renormalize=False):
        super().__init__(reduction=reduction)
        self.tasks = tasks
        self.model, self.input_shape = self.get_model(model)
        self.splits = splits
        self.polarity = polarity
        self.eps = torch.finfo(torch.get_default_dtype()).eps
        self.resize_transform = tv.transforms.Resize(self.input_shape[-2:])
        self.weights = weights or [1.0, 1.0]
        self.couple = couple 
        self.mode = mode
        self.mmd_scale = mmd_scale
        self.mmd = MMD(kernel_args={"scale": self.mmd_scale})
        self.normalize_transform = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.renormalize = renormalize

    def get_model(self, model_id: str):
        if os.path.isdir(model_id):
            model, config = get_model_from_path(model_id)
            input_shape = config.model.input_shape
            self.normalize = False
        else:
            model = getattr(tv.models, model_id)(pretrained=True)
            input_shape = (3, 299, 299) if model_id == "inception_v3" else (3, 256, 256)
            self.normalize=True
        return model.eval(), input_shape

    def get_inception_score(self, features: torch.Tensor):
        # random chunk
        idx = torch.randperm(features.shape[0])
        features = features[idx]
        # calculate probs and logits
        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)
        # split into groups
        prob = prob.chunk(self.splits, dim=0)
        log_prob = log_prob.chunk(self.splits, dim=0)
        # calculate score per split
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
        kl_ = [p * (log_p - (m_p + self.eps).log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
        kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
        kl = torch.stack(kl_)
        if torch.isnan(kl).any():
            print('nan in is!')
        return kl.mean(), kl.std()

    def get_fid(self, features_real, features_fake, tolerance=1.e-6):

        real_features_sum = features_real.sum(dim=0)
        real_features_cov_sum = features_real.t().mm(features_real)
        real_features_num_samples = features_real.shape[0]
        fake_features_sum = features_fake.sum(dim=0)
        fake_features_cov_sum = features_fake.t().mm(features_fake)
        fake_features_num_samples = features_fake.shape[0]

        mean_real = (real_features_sum / real_features_num_samples).unsqueeze(0)
        mean_fake = (fake_features_sum / fake_features_num_samples).unsqueeze(0)

        cov_real_num = real_features_cov_sum - real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (real_features_num_samples - 1)
        cov_fake_num = fake_features_cov_sum - fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (fake_features_num_samples - 1)

        mu_real = mean_real.squeeze(0)
        mu_fake = mean_fake.squeeze(0)

        diff = mu_real - mu_fake

        eps = torch.finfo(features_real.dtype).eps
        # covmean = sqrtm(cov_real.mm(cov_fake))
        # # Product might be almost singular
        # if not torch.isfinite(covmean).all():
        #     offset = torch.eye(cov_real.size(0), device=mu_real.device, dtype=mu_real.dtype) * eps
        #     covmean = sqrtm((cov_real + offset).mm(cov_fake + offset))
        # tr_covmean = torch.trace(covmean)
        # get trace directly from eigenvalues
        covmean_eig = torch.linalg.eigvalsh(cov_real.mm(cov_fake))
        idx = torch.where(covmean_eig >= tolerance)
        tr_covmean = covmean_eig[idx[0]].sqrt().sum()
        fid = diff.dot(diff) + torch.trace(cov_real) + torch.trace(cov_fake) - 2 * tr_covmean
        if torch.isnan(fid).any():
            print('nan in fid!')
        return fid

    def get_classwise_mmd(self, x:torch.Tensor, target: torch.Tensor, y_target: torch.Tensor):
        assert self.tasks is not None
        assert y_target is not None
        mmd = 0
        for t, t_config in self.tasks.items():
            meta = y_target[t]
            for i in range(t_config.dim):
                current_idx = torch.where(meta == i)[0]
                if current_idx.shape[0] == 0:
                    continue
                target_label = target[current_idx]
                mmd = mmd + self.mmd(x, target_label)
        return mmd

    def get_global_mmd(self, x:torch.Tensor, target: torch.Tensor):
        return self.mmd(x, target).mean(0)

    def get_spectral_divergence(self, x: torch.Tensor, target: torch.Tensor):
        return torchmetrics.SpectralDistortionIndex()(x, target)

    def get_lpips(self, x: torch.Tensor, target: torch.Tensor):
        return torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity

    def get_uqi(self, x: torch.Tensor, target: torch.Tensor):
        return torchmetrics.UniversalImageQualityIndex(x, target)

    def resize_data(self, x: torch.Tensor):
        if x.shape[-3] != self.input_shape[-3]:
            repeat_dims = [1,]*len(x.shape)
            repeat_dims[-3] = self.input_shape[-3]
            x = x.repeat(*repeat_dims)
        if tuple(x.shape[-2:]) != tuple(self.input_shape[-2:]):
            x = self.resize_transform(x)
        if self.renormalize:
            x = (x + 1) / 2
        if self.normalize:
            x = self.normalize_transform(x)
        return x
    
    def get_outputs(self, features):
        if isinstance(features, tv.models.inception.InceptionOutputs):
            features = features.logits
        return features

    def forward(self, x: torch.Tensor, target: torch.Tensor=None, y = None, drop_detail: bool = False, **kwargs):
        # prepare data
        self.model.eval()
        x = self.resize_data(x)
        if next(self.model.parameters()).device != x.device:
            self.model = self.model.to(x.device).eval()
        # prepare features
        gen_loss = None
        if isinstance(self.model, GAN):
            features = self.model.discriminator(x)
            if "gan_loss" in self.weights:
                gen_loss = self.model.generator_loss(self.model.generator, None, None, features)
            features = features['domain']
        else:
            features = self.get_outputs(self.model(x))
        if self.couple in ["is/fid", "mmd"]:
            with torch.no_grad():
                target = self.resize_data(target)
                features_real = self.get_outputs(self.model(target))
        # compute losses
        loss = 0
        losses = {}
        if self.couple == "is/fid":
            kl_mean, kl_std = self.get_inception_score(features)
            fid = self.get_fid(features_real, features)
            losses = {'is_mean': kl_mean.detach().cpu(), 'is_std': kl_std.detach().cpu(), 'fid': fid.detach().cpu()}
            if self.mode == "sum":
                loss = self.weights[0] * kl_mean + self.weights[1] * fid
            elif self.mode == "ratio":
                loss = (self.weights[0] * fid) * (self.weights[1] * kl_mean + eps)
        elif self.couple == "mmd":
            class_mmd = self.get_classwise_mmd(features, features_real, y)
            global_mmd = self.get_global_mmd(features, features_real)
            losses = {'class_mmd': class_mmd.detach().cpu(), 'global_mmd': global_mmd.detach().cpu()}
            if self.mode == "sum":
                loss = self.weights[0] * class_mmd + self.weights[1] * global_mmd
            elif self.mode == "ratio":
                loss = class_mmd.pow(self.weights[0]) / global_mmd.pow(self.weights[1])
        # if 'spec_div' in self.weights:
        #     spec_div = self.get_spectral_divergence(x, target)
        #     loss = loss + self.weights['spec_div'] * spec_div
        #     losses = {**losses, 'spec_div': spec_div.detach().cpu()}
        # if 'lpips' in self.weights:
        #     lpips = self.get_spectral_divergence(x, target)
        #     loss = loss + self.weights['lpips'] * lpips
        #     losses = {**losses, 'lpips': lpips.detach().cpu()}
        # if 'uqi' in self.weights:
        #     uqi = self.get_uqi(x, target)
        #     loss = loss + self.weights['uqi'] * uqi
        #     losses = {**losses, 'uqi': uqi.detach().cpu()}
        # if "gan_loss" in self.weights:
        #     assert gen_loss is not None, "disc_loss only work with Discriminator"
        #     loss = loss + self.weights['gan_loss'] * gen_loss 
        #     losses = {**losses, 'gan_loss': gen_loss.detach().cpu()}
        if drop_detail:
            return loss, losses
        else:
            return loss

        