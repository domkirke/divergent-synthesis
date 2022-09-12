from collections import namedtuple
from active_divergence.data.audio import module
import torch, torch.nn as nn, numpy as np, pdb
import torch.distributions as dist
from active_divergence.utils import checklist, checktuple, print_stats, checkdist, parse_slice, frame
from omegaconf import OmegaConf
from active_divergence.modules import layers as layers, mlp_dist_hash, conv_dist_hash, Reshape
from typing import ForwardRef, Tuple, Union


class WaveNet(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.input_channels = config.input_channels
        self.n_blocks = config.get('n_layers', 4)
        self.bias = config.get('bias', False)
        self.dilation_channels = config.get('dilation_channels', 32)
        self.residual_channels = config.get('residual_channels', self.dilation_channels)
        self.skip_channels = config.get('skip_channels', 32)
        self.end_channels = config.get('end_channels', self.skip_channels)
        self.nnlin = config.get('nnlin', 'ReLU')
        self.layer = config.get('layer', 'WaveNetBlock')
        self.block_args = checklist(config.get("block_args", {}), n=self.n_blocks)
        self.init_modules()

    @property
    def input_block_size(self):
        """length required to train the model"""
        size = 0
        for m in self.block_args:
            size  += m['dilation_rate'] ** m['n_convs_per_block']
        return size

    @property
    def min_input_block_size(self):
        size = 0
        for m in self.block_args:
            size  += m['dilation_rate'] ** m['n_convs_per_block']
        return size

    @property
    def generation_lookahead(self):
        return 1

    @property
    def context_block_size(self):
        """length required to generate from module"""
        return None

    def init_modules(self):
        blocks = []
        layer = getattr(layers, self.layer)
        self.input_conv = nn.Conv1d(self.input_channels, self.residual_channels, 1, bias=self.bias)
        for n in range(self.n_blocks):
            blocks.append(layer(dilation_channels = self.dilation_channels, 
                                residual_channels = self.residual_channels,
                                skip_channels = self.skip_channels,
                                bias = self.bias,
                                **dict(self.block_args[n])))
        self.blocks = nn.ModuleList(blocks)
        self.final_conv = nn.Sequential(getattr(nn, self.nnlin)(), 
                                        nn.Conv1d(self.skip_channels, self.end_channels, 1),
                                        getattr(nn, self.nnlin)(),
                                        nn.Conv1d(self.end_channels, self.input_channels, 1), 
                                        nn.Softmax(dim=-2))
        
    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.one_hot(x.long(), self.input_channels)
        x = x.transpose(-2, -1).float().contiguous()
        skip_buffer = None
        residual_in = self.input_conv(x)
        previous_out = residual_in
        for n in range(self.n_blocks):
            previous_out, skip_buffer = self.blocks[n](previous_out, skip_buffer)
        out = self.final_conv(skip_buffer)
        out = dist.Categorical(out.transpose(-2, -1))
        return out


class SampleRNN(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.input_size = config.input_size
        self.output_size = config.get('output_dim', self.input_size)
        self.n_tiers = config.get('n_tiers', 3)
        self.dim = checklist(config.get('dim'), n=self.n_tiers)
        self.expand_factor = config.get('expand_factor', 4)
        self.block_args = checklist(config.get('block_args'), n=self.n_tiers)
        self.recurrent_type = config.get('recurrent_type', "GRU")
        self.linear_type = config.get('linear_type', "MLP")
        self.target_dist = config.get('target_dist')
        self.init_modules()

    @property
    def input_block_size(self):
        return self.context_block_size * 2

    @property
    def context_block_size(self):
        return int(self.expand_factor ** (self.n_tiers - 1)) 

    @property
    def generation_lookahead(self):
        return self.expand_factor

    def init_modules(self):
        rnn_modules = []
        input_linear_modules = []
        between_linear_modules = []
        for n in range(self.n_tiers):
            # set up recurrent modules
            recurrent_type = getattr(layers, self.recurrent_type)
            if n == 0:
                input_samples = (self.context_block_size, self.input_size)
            else:
                input_samples = self.dim[n]
            current_rnn = recurrent_type(input_samples, self.dim[n], **dict(self.block_args[n]))
            rnn_modules.append(current_rnn)
            # set up linear modules between tiers
            if n < self.n_tiers - 1:
                linear_type = getattr(layers, self.linear_type)
                current_linear = linear_type(self.dim[n+1], self.dim[n+1], bias=False, nlayers=0)
                between_linear_modules.append(current_linear)
            # set up input linear modules
            if n > 0:
                linear_type = getattr(layers, self.linear_type) 
                current_exp = max(1, self.n_tiers - (n + 1))
                current_linear = linear_type((self.expand_factor ** current_exp, self.input_size), self.dim[n], nlayers=0, bias=False)
                input_linear_modules.append(current_linear)
        self.rnn_modules = nn.ModuleList(rnn_modules)
        self.input_linear_modules = nn.ModuleList(input_linear_modules)
        self.between_linear_modules = nn.ModuleList(between_linear_modules)
        self.final_module = layers.MLP(self.dim[-1], self.output_size, nlayers=0, bias=False)
        
        if self.target_dist is not None:
            self.target_dist = checkdist(self.target_dist)
            self.dist_module = mlp_dist_hash[self.target_dist]()
        else:
            self.dist_module = nn.Softmax(-1)

    def forward(self, x, **kwargs):
        #TODO fix frame (removes last!)
        x = torch.nn.functional.one_hot(x.long(), self.input_size).float()
        x = torch.stack([frame(x_tmp, self.input_block_size, self.context_block_size, dim=0) for x_tmp in x])
        # top-tier activations
        input_h = x[..., :self.context_block_size, :].reshape(x.shape[0], x.shape[1], -1)
        h = self.rnn_modules[0](input_h, return_hidden=False).unsqueeze(-2)

        for n in range(1, self.n_tiers):
            h = self.between_linear_modules[n-1](h)
            x_t = x.view(np.prod(x.shape[:2]), x.shape[-2], x.shape[-1])
            current_context_size = self.expand_factor ** max(self.n_tiers - (n + 1), 1)
            current_hop_size = self.expand_factor ** max(self.n_tiers - (n + 1), 0)
            x_t = torch.stack([frame(x_tmp, current_context_size, current_hop_size, dim=0) for x_tmp in x_t])
            t_offset = int((self.input_block_size / current_hop_size) / 2 - current_context_size / current_hop_size)
            x_t = x_t[..., t_offset:, :, :]
            x_t = self.input_linear_modules[n-1](x_t)
            x_t = x_t.reshape(x.shape[0], x.shape[1], *x_t.shape[1:])
            h_t = h.repeat_interleave(self.expand_factor, dim=-2)
            input_h = x_t + h_t
            h = self.rnn_modules[n](input_h.reshape(x.shape[0], -1, x_t.shape[-1]), return_hidden=False)
            h = h.reshape(*x.shape[:2], -1, h.shape[-1])


        out = self.final_module(h)
        out = out.view(x.shape[0], out.shape[-3]*out.shape[-2], out.shape[-1])
        out = self.dist_module(out)
        return out



