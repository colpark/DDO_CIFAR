import math
import functools
import numpy as np
import torch
import torch.nn as nn
import torch_geometric as tg
import torch_geometric.nn as tgnn


def get_mgrid(img_height, dim=2):
    grid = torch.linspace(0, img_height-1, img_height) / img_height
    if dim == 1:
        grid = grid[None, None]
    elif dim == 2:
        grid = torch.cat([grid[None,None,...,None].repeat(1, 1, 1, img_height),
                          grid[None,None,None].repeat(1, 1, img_height, 1)], dim=1)
    elif dim == 3:
        grid = torch.cat([grid[None,None,...,None,None].repeat(1, 1, 1, img_height, img_height),
                          grid[None,None,None,...,None].repeat(1, 1, img_height, 1, img_height),
                          grid[None,None,None,None].repeat(1, 1, img_height, img_height, 1)], dim=1)
    else:
        raise NotImplementedError
    return grid

def get_batch(batch_size, num_points):
    return torch.arange(batch_size).reshape(batch_size, 1).expand(batch_size, num_points).reshape(-1)

def get_coords(batch_size, height):
    v = get_mgrid(height, dim=2).repeat(batch_size, 1, 1, 1)
    v = v.permute(0, 2, 3, 1).reshape(batch_size, height*height, 2)
    return v


def knn_graph(pos_src, pos_dst, k:int, batch_src=None, batch_dst=None, flow:str='source_to_target'):
    edge_index = tgnn.knn(pos_src, pos_dst, k=k, batch_x=batch_src, batch_y=batch_dst)
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]
    return torch.stack([row, col], dim=0)

def radius_graph(pos_src, pos_dst, r:float, batch_src=None, batch_dst=None, flow:str='source_to_target', max_num_neighbors=1000):
    edge_index = tgnn.radius(pos_src, pos_dst, r=r, batch_x=batch_src, batch_y=batch_dst, max_num_neighbors=max_num_neighbors)
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]
    return torch.stack([row, col], dim=0)

def get_relative_position(pos_src, pos_dst, edge_index):
    src = edge_index[0]
    dst = edge_index[1]
    return pos_dst.index_select(0, dst) - pos_src.index_select(0, src)

def get_edge_index_attr(pos_src, batch_src, pos_dst, batch_dst, radius:float=None, k:int=None):
    if radius is not None:
        edge_index = radius_graph(pos_src, pos_dst, batch_src=batch_src, batch_dst=batch_dst, r=radius, flow='source_to_target')
    elif k is not None:
        edge_index = knn_graph(pos_src, pos_dst, batch_src=batch_src, batch_dst=batch_dst, k=k, flow='source_to_target')
    edge_attr = get_relative_position(pos_src, pos_dst, edge_index)
    return edge_index, edge_attr


def _turkey(x, alpha=0.2, beta=0.25, mul=1., **kwargs):
    alpha = mul * alpha
    beta = mul * beta
    assert beta >= alpha
    def _cos(x):
        return 0.5 * (1 + torch.cos( 2. * np.pi * x / alpha))
    mask0 = ( - 0.5 * beta <= x ).float() * (  x <= - 0.5 * (beta - alpha) ).float()
    mask1 = ( - 0.5 * (beta - alpha) < x ).float() * ( x <= 0.5 * (beta - alpha) ).float()
    mask2 = ( 0.5 * (beta - alpha) < x ).float() * (  x <= 0.5 * beta ).float()
    return  mask0 * _cos( x + 0.5 * (beta - alpha) ) + mask1 + mask2 * _cos( x - 0.5 * (beta - alpha) )

def _normal(x, std=0.1, mul=1., **kwargs):#0.025):
    std = mul * std
    C = 1. #/ ( std * np.sqrt(2. * np.pi) )
    return C * torch.exp( - 0.5 * x**2  / std**2 )

def _gabor(x, a=0.025, k0=40., mul=1., **kwargs):
    raise NotImplementedError
    C = 1. #/ ( a * np.sqrt(2. * np.pi) * np.exp( - 0.5 * a**2 * k0**2 ) )
    return C * torch.exp( - 0.5 * x**2 / a**2 ) * torch.cos(k0 * x)

def _triangle(x, beta=0.25, mul=1., **kwargs):
    beta = mul * beta
    slope = 2. / beta
    mask0 = (-beta <= x).float() * (x <= 0).float()
    mask1 = (0 < x).float() * (x <= beta).float()
    return slope * (x + beta) * mask0 - slope * (x - beta) * mask1



def turkey1d(x, *args, **kwargs):
    return _turkey(x[:,0:1], *args, **kwargs)

def triangle1d(x, *args, **kwargs):
    return _triangle(x[:,0:1], *args, **kwargs)


def turkey2d(x, *args, **kwargs):
    return _turkey(x[:,0:1], *args, **kwargs) * _turkey(x[:,1:2], *args, **kwargs)

def triangle2d(x, *args, **kwargs):
    return _triangle(x[:,0:1], *args, **kwargs) * _triangle(x[:,1:2], *args, **kwargs)


def turkey3d(x, *args, **kwargs):
    return _turkey(x[:,0:1], *args, **kwargs) * _turkey(x[:,1:2], *args, **kwargs) * _turkey(x[:,2:3], *args, **kwargs)

def triangle3d(x, *args, **kwargs):
    return _triangle(x[:,0:1], *args, **kwargs) * _triangle(x[:,1:2], *args, **kwargs) * _triangle(x[:,2:3], *args, **kwargs)


def triangle(x, *args, **kwargs):
    pos_dim = x.shape[-1]
    if pos_dim == 1:
        return triangle1d(x, *args, **kwargs)
    elif pos_dim == 2:
        return triangle2d(x, *args, **kwargs)
    elif pos_dim == 3:
        return triangle3d(x, *args, **kwargs)
    else:
        raise NotImplementedError

def turkey(x, *args, **kwargs):
    pos_dim = x.shape[-1]
    if pos_dim == 1:
        return turkey1d(x, *args, **kwargs)
    elif pos_dim == 2:
        return turkey2d(x, *args, **kwargs)
    elif pos_dim == 3:
        return turkey3d(x, *args, **kwargs)
    else:
        raise NotImplementedError


def get_window_fn(window, **kwargs):
    if window.startswith('normal'):
        window_fn = functools.partial(normal, **kwargs)
    elif window.startswith('turkey'):
        window_fn = functools.partial(turkey, **kwargs)
    elif window.startswith('gabor'):
        window_fn = functools.partial(gabor, **kwargs)
    elif window.startswith('triangle'):
        window_fn = functools.partial(triangle, **kwargs)
    else:
        raise ValueError
    return window_fn


from typing import Callable, Optional, Union

from torch import Tensor

from torch.nn.parameter import Parameter

#import torch_geometric.nn as tgnn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor


class Sine(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class Scale(nn.Module):
    def __init__(self, val=100.):
        super().__init__()
        self.val = val
        self.scale = Parameter(torch.ones(1)*val)
        # self.register_buffer('scale', torch.empty(1, 1))
        # torch.nn.init.constant_(self.scale, val=val)

    def forward(self, x):
        # n = x.ndim
        # if n == 3:
        #     scale = self.scale[...,None]
        # elif n == 4:
        #     scale = self.scale[...,None,None]
        # elif n == 5:
        #     scale = self.scale[...,None,None,None]
        # else:
        #     raise NotImplementedError
        return self.scale * x

    def extra_repr(self) -> str:
        s = f'init={self.val}'
        return s


class BaseContinuousConv(MessagePassing):
    def __init__(self,
                 pos_dim: int,
                 base_resolution: int,
                 in_channels: int,
                 out_channels: int,
                 fc_channels: int,
                 kernel_size: int,
                 groups: int = 1,
                 bias: bool = True,
                 aggr: str = 'mean',
                 act: nn.Module = nn.SiLU(),
                 use_weight_net: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)

        assert in_channels % groups == 0, 'in_channels: {:d}, groups: {:d}'.format(in_channels, groups)
        assert out_channels % groups == 0, 'out_channels: {:d}, groups: {:d}'.format(out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.pos_dim = pos_dim
        self.base_resolution = base_resolution
        self.fc_channels = fc_channels
        self.kernel_size = kernel_size

        # init member variables
        assert kernel_size % 2 == 1, f"kernel_size must be odd number. kernel_size: {kernel_size}"
        even_kernel_size = kernel_size + 1
        beta = float(even_kernel_size//2) / self.base_resolution
        self.radius = beta #* np.sqrt(2)

        self.window_fn = get_window_fn('turkey', alpha=beta*2, beta=beta*2)
        # self.window_fn = get_window_fn('triangle', beta=beta)

        # weight and bias
        if use_weight_net:
            self.weight_net = nn.Sequential(
                Scale(val=100.),
                nn.Linear(pos_dim, fc_channels),
                nn.BatchNorm1d(fc_channels),
                Sine(),
                nn.Linear(fc_channels, fc_channels),
                nn.BatchNorm1d(fc_channels),
                Sine(),
                nn.Linear(fc_channels, out_channels*in_channels//groups),
            )
        else:
            assert out_channels == in_channels
            self.groups = groups = in_channels
            self.weight_net = None
        if bias:
            bsz = [1, out_channels] + [1]*pos_dim
            self.bias = Parameter(torch.empty(*bsz))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self,
                x: Union[OptTensor, PairOptTensor],
                edge_index: Adj,
                edge_attr: OptTensor = None,
                ) -> Tensor:
        """"""
        if not isinstance(x, tuple):
            x: PairOptTensor = (x, None)

        # propagate_type: (x: PairOptTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index,
                             x=x,
                             edge_attr=edge_attr,
                             size=None,
                             )

        # bias
        if self.bias is not None:
            out = out + self.bias[None]

        return out

    def message(self, x_j: Tensor, index: Tensor, edge_attr: OptTensor) -> Tensor:
        """
        x_j:       input values at input indices
        index:     query indices
        edge_attr: from-source-to-query edge attributes (i -> j)
        """
        # init
        npt = x_j.shape[0]

        if self.weight_net is not None:
            # get weight
            w = self.window_fn(edge_attr) # n x 1
            weight = w * self.weight_net(edge_attr) # n x c_out*c_in

            # reshape
            weight = weight.reshape(
                npt, self.groups, self.out_channels//self.groups, self.in_channels//self.groups)
            x_j = x_j.reshape(
                npt, self.groups, 1, self.in_channels//self.groups)

            # forward
            msg = (weight*x_j).sum(dim=-1).reshape(npt, self.out_channels)

        else:
            # get weight
            weight = self.window_fn(edge_attr) # n x 1

            # reshape
            weight = weight.reshape(npt, 1, 1, 1).expand(npt, self.groups, self.out_channels//self.groups, self.in_channels//self.groups)
            x_j = x_j.reshape(
                npt, self.groups, 1, self.in_channels//self.groups)

            # forward
            msg = (weight*x_j).sum(dim=-1).reshape(npt, self.out_channels)

        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'\n  bs = {self.base_resolution}, '
                f'\n  in_channels = {self.in_channels}, '
                f'\n  out_channels = {self.out_channels}, '
                f'\n  fc_channels = {self.fc_channels}, '
                f'\n  kernel_size = {self.kernel_size}, '
                f'\n  groups = {self.groups}, '
                f'\n  bias = {self.bias is not None}, '
                f'\n  weight_net = {self.weight_net})'
               )

class ContinuousConv(BaseContinuousConv):
    def __init__(self,
                 pos_dim: int,
                 base_resolution: int,
                 in_channels: int,
                 out_channels: int,
                 *args,
                 embed_channels: int = None,
                 use_act_embd: bool = True,
                 act: nn.Module = nn.SiLU(),
                 **kwargs,
                 ):
        super().__init__(
                pos_dim,
                base_resolution,
                embed_channels if embed_channels is not None else in_channels,
                embed_channels if embed_channels is not None else out_channels,
                *args,
                act=act,
                **kwargs,
                )

        self.embed_channels = embed_channels
        if embed_channels is not None:
            if use_act_embd:
                self.enc = nn.Sequential(
                        act,
                        nn.Linear(in_channels, embed_channels),
                        )
            else:
                self.enc = nn.Linear(in_channels, embed_channels)
            self.dec = nn.Linear(embed_channels, out_channels)
        else:
            self.enc = None
            self.dec = None

    def forward(self, *args, **kwargs) -> Tensor:
        out = super().forward(*args, **kwargs)
        if self.dec is not None:
            out = self.dec(out)
        return out

    def message(self, x_j: Tensor, index: Tensor, edge_attr: OptTensor) -> Tensor:
        if self.enc is not None:
            x_j = self.enc(x_j)
        msg = super().message(x_j, index, edge_attr)
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'\n  bs = {self.base_resolution}, '
                f'\n  in_channels = {self.in_channels}, '
                f'\n  out_channels = {self.out_channels}, '
                f'\n  embed_channels = {self.embed_channels}, '
                f'\n  fc_channels = {self.fc_channels}, '
                f'\n  kernel_size = {self.kernel_size}, '
                f'\n  groups = {self.groups}, '
                f'\n  bias = {self.bias is not None}, '
                f'\n  weight_net = {self.weight_net})'
               )


class ContinuousConv1d(ContinuousConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pos_dim=1, **kwargs)


class ContinuousConv2d(ContinuousConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pos_dim=2, **kwargs)
