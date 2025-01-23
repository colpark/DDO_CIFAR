'''
Copied and modified from https://github.com/samb-t/infty-diff/blob/main/models/model_sparse.py
'''
from typing import Callable, Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import calculate_gain
from einops import rearrange, repeat
from .contconv import get_coords, get_batch, get_edge_index_attr, ContinuousConv2d


# spconv
try:
    import spconv as spconv_core
    spconv_core.constants.SPCONV_ALLOW_TF32 = True
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor
    SPCONV_AVAILABLE = True
except Exception:
    SPCONV_AVAILABLE = False

# torchsparse
try:
    import torchsparse
    from torchsparse import nn as spnn
    from torchsparse import SparseTensor
    TORCHSPARSE_AVAILABLE = True
except Exception:
    TORCHSPARSE_AVAILABLE = False

# minkowski
try:
    import MinkowskiEngine as ME
    MINKOWSKI_AVAILABLE = True
except Exception:
    MINKOWSKI_AVAILABLE = False

# torch_geometric
try:
    import torch_geometric
    from torch_geometric import nn as tgnn
    TORCHGEOMETRIC_AVAILABLE = True
except Exception:
    TORCHGEOMETRIC_AVAILABLE = False


class TorchGeometricTensor:
    def __init__(
        self,
        feats: torch.Tensor,
        batch: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> None:
        self.feats = feats
        self.batch = batch
        self.coords = coords
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    @property
    def F(self) -> torch.Tensor:
        return self.feats

    @F.setter
    def F(self, feats: torch.Tensor) -> None:
        self.feats = feats

    @property
    def C(self) -> torch.Tensor:
        return self.coords

    @C.setter
    def C(self, coords: torch.Tensor) -> None:
        self.coords = coords

    def update_edges(self, radius) -> None:
        self.edge_index, self.edge_attr = get_edge_index_attr(self.coords, self.batch, self.coords, self.batch, radius=radius)

    def cpu(self):
        self.feats = self.feats.cpu()
        self.batch = self.batch.cpu()
        if self.coords is not None:
            self.coords = self.coords.cpu()
        if self.edge_index is not None:
            self.edge_index = self.edge_index.cpu()
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.cpu()
        return self

    def cuda(self):
        self.feats = self.feats.cuda()
        self.batch = self.batch.cuda()
        if self.coords is not None:
            self.coords = self.coords.cuda()
        if self.edge_index is not None:
            self.edge_index = self.edge_index.cuda()
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.cuda()
        return self

    def half(self):
        self.feats = self.feats.half()
        return self

    def detach(self):
        self.feats = self.feats.detach()
        self.batch = self.batch.detach()
        if self.coords is not None:
            self.coords = self.coords.detach()
        if self.edge_index is not None:
            self.edge_index = self.edge_index.detach()
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.detach()
        return self

    def to(self, device, non_blocking: bool = True):
        self.feats = self.feats.to(device, non_blocking=non_blocking)
        self.batch = self.batch.to(device, non_blocking=non_blocking)
        if self.coords is not None:
            self.coords = self.coords.to(device, non_blocking=non_blocking)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device, non_blocking=non_blocking)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device, non_blocking=non_blocking)
        return self

    def __add__(self, other):
        output = TorchGeometricTensor(
            feats=self.feats + other.feats,
            batch=self.batch,
            coords=self.coords,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
        )
        return output

"""
Wrapper around the different backends
"""
# TODO: At test time a dense conv can be used instead
class SparseConvResBlock(nn.Module):
    def __init__(self, img_size, embed_dim, fc_channels=32, kernel_size=7, mult=2, skip_dim=None, time_emb_dim=None,
                 epsilon=1e-5, z_dim=None, backend="torch_dense", depthwise=False, **kwargs):
        super().__init__()
        self.backend = backend
        if self.backend == "spconv":
            assert SPCONV_AVAILABLE, "spconv backend is not detected."
            block = SPConvResBlock
        elif self.backend == "torchsparse":
            assert TORCHSPARSE_AVAILABLE, "torchsparse backend is not detected."
            block = TorchsparseResBlock
        elif self.backend == "minkowski":
            assert MINKOWSKI_AVAILABLE, "Minkowski Engine backend is not detected."
            block =  MinkowskiConvResBlock
        elif self.backend == "torch_geometric":
            assert TORCHGEOMETRIC_AVAILABLE, "Torch geometric backend is not detected."
            block = TorchGeometricResBlock
        elif self.backend == "torch_dense":
            block = TorchDenseConvResBlock
        else:
            raise Exception("Unrecognised backend.")

        self.block = block(img_size, embed_dim, fc_channels=fc_channels, kernel_size=kernel_size, mult=mult, skip_dim=skip_dim,
                 time_emb_dim=time_emb_dim, epsilon=epsilon, z_dim=z_dim, depthwise=depthwise, **kwargs)

    @property
    def radius(self):
        return self.block.radius

    def get_normalising_conv(self):
        return self.block.get_normalising_conv()

    def forward(self, x, t=None, skip=None, z=None, norm=None):
        if isinstance(x, torch.Tensor) and len(x.shape) == 4 and self.backend != "torch_dense":
            # If image shape passed in then use more efficient dense convolution
            return self.block.dense_forward(x, t=t, skip=skip, z=z, norm=norm)
        elif isinstance(x, torch.Tensor) and len(x.shape) == 4 and self.backend == "torch_dense" and not isinstance(norm, tuple):
            # if backend is torch_dense and we input the norm is not a tuple then we can run in dense mode.
            return self.block.dense_forward(x, t=t, skip=skip, z=z, norm=norm)
        else:
            return self.block(x, t=t, skip=skip, z=z, norm=norm)

class TorchGeometricResBlock(nn.Module):
    def __init__(self, img_size, embed_dim, fc_channels=32, kernel_size=7, mult=2, skip_dim=None, time_emb_dim=None,
                 epsilon=1e-5, z_dim=None, depthwise=True, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.spatial_size = img_size ** 2
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        self.embed_dim = embed_dim
        self.groups = embed_dim if depthwise else 1

        if skip_dim is not None:
            self.skip_linear = nn.Linear(embed_dim + skip_dim, embed_dim)

        # TODO: check where is best to have the 1 dimension.
        self.norm1 = nn.LayerNorm(embed_dim)
        self.conv = ContinuousConv2d(
            base_resolution=img_size,
            in_channels=embed_dim,
            out_channels=embed_dim,
            embed_channels=embed_dim if depthwise else None,
            fc_channels=fc_channels,
            kernel_size=kernel_size,
            groups=self.groups,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*mult),
            nn.GELU(),
            nn.Linear(embed_dim*mult, embed_dim)
        )

        self.time_mlp1, self.time_mlp2, self.z_mlp1, self.z_mlp2 = None, None, None, None
        if time_emb_dim is not None:
            self.time_mlp1 = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, embed_dim*2)
            )
            self.time_mlp2 = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, embed_dim*2)
            )
        if z_dim is not None:
            self.z_mlp1 = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.z_mlp2 = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )

    @property
    def radius(self):
        return self.conv.radius

    def modulate(self, h, t=None, z=None, norm=None, t_mlp=None, z_mlp=None):
        if isinstance(h, TorchGeometricTensor):
            feats = h.feats
        else:
            feats = h
        feats = norm(feats)
        q_sample = feats.size(0) // t.size(0)
        if t is not None:
            t = t_mlp(t)
            t = repeat(t, "b c -> (b l) c", l=q_sample)
            t_scale, t_shift = t.chunk(2, dim=-1)
            feats = feats * (1 + t_scale) + t_shift
        if z is not None:
            z_scale = z_mlp(z)
            z_scale = repeat(z_scale, "b c -> (b l) c", l=q_sample)
            feats = feats * (1 + z_scale)
        if isinstance(h, TorchGeometricTensor):
            h = convert_to_backend_form_like(feats, h, backend="torch_geometric", rearrange_x=False)
        else:
            h = feats
        return h

    def forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, TorchGeometricTensor)
        if x.edge_index is None:
            x.update_edges(self.radius)

        # Skip connection
        if skip is not None:
            feats = torch.cat((x.feats, skip.feats), dim=-1)
            feats = self.skip_linear(feats)
            x = convert_to_backend_form_like(feats, x, backend="torch_geometric", rearrange_x=False)

        h = x
        if t is not None or z is not None:
            h = self.modulate(h, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1)

        h.feats = self.conv(h.feats, h.edge_index, h.edge_attr)
        if norm is not None:
            h = tg_div(h, norm)
        x = tg_add(x, h)

        if t is not None or z is not None:
            h = self.modulate(x, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2)
        x = tg_add(x, self.mlp(h.feats))

        return x

class TorchsparseResBlock(nn.Module):
    def __init__(self, img_size, embed_dim, kernel_size=7, mult=2, skip_dim=None, time_emb_dim=None,
                 epsilon=1e-5, z_dim=None, **kwargs): #True):
        super().__init__()
        self.img_size = img_size
        self.spatial_size = img_size ** 2
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        self.embed_dim = embed_dim
        # self.groups = embed_dim if depthwise else 1

        if skip_dim is not None:
            self.skip_linear = nn.Linear(embed_dim + skip_dim, embed_dim)

        # TODO: check where is best to have the 1 dimension.
        self.norm1 = nn.LayerNorm(embed_dim)
        self.conv = spnn.Conv3d(embed_dim, embed_dim, kernel_size=(1,kernel_size,kernel_size), bias=False) #depthwise=depthwise)
        self._custom_kaiming_uniform_(self.conv.kernel, a=math.sqrt(5))

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*mult),
            nn.GELU(),
            nn.Linear(embed_dim*mult, embed_dim)
        )

        self.time_mlp1, self.time_mlp2, self.z_mlp1, self.z_mlp2 = None, None, None, None
        if time_emb_dim is not None:
            self.time_mlp1 = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, embed_dim*2)
            )
            self.time_mlp2 = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, embed_dim*2)
            )
        if z_dim is not None:
            self.z_mlp1 = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.z_mlp2 = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )

    def get_torch_kernel(self, img_size, round_down=True):
        if img_size != self.img_size:
            ratio = img_size / self.img_size
            new_kernel_size = self.kernel_size * ratio
            if round_down:
                new_kernel_size = 2 * round((new_kernel_size - 1) / 2) + 1
            else:
                new_kernel_size = math.floor(new_kernel_size / 2) * 2 + 1
            new_kernel_size = max(new_kernel_size, 3)
            kernel = rearrange(self.conv.kernel, "(h w) i o -> o i w h", h=self.kernel_size)
            kernel = F.interpolate(kernel, size=new_kernel_size, mode="bilinear")
            return kernel
        else:
            return rearrange(self.conv.kernel, "(h w) i o -> o i w h", h=self.kernel_size)

    def dense_forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, torch.Tensor), "Dense forward expects x to be a torch Tensor"
        assert len(x.shape) == 4, "Dense forward expects x to be 4D: (b, c, h, w)"

        # Skip connection
        batch_size, height, width = x.size(0), x.size(2), x.size(3)
        h = rearrange(x, "b c h w -> (b h w) c")
        if skip is not None:
            skip = rearrange(skip, "b c h w -> (b h w) c")
            h = torch.cat((h, skip), dim=-1)
            h = self.skip_linear(h)
        x = rearrange(h, "(b h w) c -> b c h w", b=batch_size, h=height, w=width)

        if t is not None or z is not None:
            h = self.modulate(h, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1)
        h = rearrange(h, "(b h w) c -> b c h w", b=batch_size, h=height, w=width)

        # Conv and norm
        kernel = self.get_torch_kernel(height)
        h = F.conv2d(h, kernel, padding='same') #groups=self.groups)
        h = h / norm
        x = x + h

        # elementwise MLP
        h = rearrange(x, "b c h w -> (b h w) c")
        if t is not None or z is not None:
            h = self.modulate(h, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2)
        h = self.mlp(h)
        h = rearrange(h, "(b h w) c -> b c h w", b=batch_size, h=height, w=width)

        x = x + h

        return x

    def _custom_kaiming_uniform_(self, tensor, a=0, nonlinearity='leaky_relu'):
        fan = self.embed_dim * (self.kernel_size ** 2)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(
            3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    def modulate(self, h, t=None, z=None, norm=None, t_mlp=None, z_mlp=None):
        if isinstance(h, torchsparse.SparseTensor):
            feats = h.feats
        else:
            feats = h
        feats = norm(feats)
        q_sample = feats.size(0) // t.size(0)
        if t is not None:
            t = t_mlp(t)
            t = repeat(t, "b c -> (b l) c", l=q_sample)
            t_scale, t_shift = t.chunk(2, dim=-1)
            feats = feats * (1 + t_scale) + t_shift
        if z is not None:
            z_scale = z_mlp(z)
            z_scale = repeat(z_scale, "b c -> (b l) c", l=q_sample)
            feats = feats * (1 + z_scale)
        if isinstance(h, torchsparse.SparseTensor):
            h = convert_to_backend_form_like(feats, h, backend="torchsparse", rearrange_x=False)
        else:
            h = feats
        return h

    def forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, torchsparse.SparseTensor)

        # Skip connection
        if skip is not None:
            feats = torch.cat((x.feats, skip.feats), dim=-1)
            feats = self.skip_linear(feats)
            x = convert_to_backend_form_like(feats, x, backend="torchsparse", rearrange_x=False)

        h = x
        if t is not None or z is not None:
            h = self.modulate(h, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1)

        h = self.conv(h)
        h = ts_div(h, norm)
        x = ts_add(x, h)

        if t is not None or z is not None:
            h = self.modulate(x, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2)
        x = ts_add(x, self.mlp(h.feats))

        return x

class TorchDenseConvResBlock(nn.Module):
    def __init__(self, img_size, embed_dim, kernel_size=7, mult=2, skip_dim=None, time_emb_dim=None,
                 epsilon=1e-5, z_dim=None, depthwise=False, **kwargs): #True):
        super().__init__()
        self.img_size = img_size
        self.spatial_size = img_size ** 2
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        self.groups = embed_dim if depthwise else 1

        if skip_dim is not None:
            self.skip_linear = nn.Conv2d(embed_dim + skip_dim, embed_dim, 1)

        # TODO: Try using bias
        self.norm1 = ImageLayerNorm(embed_dim)
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size, stride=1, padding='same', groups=self.groups, bias=False)

        self.norm2 = ImageLayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim*mult, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim*mult, embed_dim, 1)
        )
        self.time_mlp1, self.time_mlp2, self.z_mlp1, self.z_mlp2 = None, None, None, None
        if time_emb_dim is not None:
            self.time_mlp1 = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, embed_dim*2)
            )
            self.time_mlp2 = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, embed_dim*2)
            )
        if z_dim is not None:
            self.z_mlp1 = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.z_mlp2 = nn.Sequential(
                nn.Linear(z_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )

    def dense_forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, torch.Tensor)

        if skip is not None:
            x = torch.cat((x, skip), dim=1)
            x = self.skip_linear(x)

        h = self.modulate(x, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1)

        # Conv and norm
        h = self.conv(h)
        h = h / norm
        x = x + h

        h = self.modulate(x, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2)
        x = x + self.mlp(h)

        return x

    def modulate(self, h, t=None, z=None, norm=None, t_mlp=None, z_mlp=None):
        h = norm(h)
        if t is not None:
            t = t_mlp(t)[:,:,None,None]
            t_scale, t_shift = t.chunk(2, dim=1)
            h = h * (1 + t_scale) + t_shift
        if z is not None:
            z_scale = z_mlp(z)[:,:,None,None]
            h = h * (1 + z_scale)
        return h

    def forward(self, x, t=None, skip=None, z=None, norm=None):
        assert isinstance(x, torch.Tensor)

        # For dense conv we also pass in a mask to make the output 'sparse' again
        norm, mask = norm

        if skip is not None:
            x = torch.cat((x, skip), dim=1)
            x = self.skip_linear(x)
            x = x * mask

        h = self.modulate(x, t=t, z=z, norm=self.norm1, t_mlp=self.time_mlp1, z_mlp=self.z_mlp1)
        h =  h * mask

        # Assume that x already has 0s for missing pixels so no masking here
        h = self.conv(h)
        # NOTE: although norm is dense and not masked, everything afterwards is elementwise so this is fine
        h = h / norm
        x = x + h

        h = self.modulate(x, t=t, z=z, norm=self.norm2, t_mlp=self.time_mlp2, z_mlp=self.z_mlp2)
        x = x + self.mlp(h)

        # mask out
        return x * mask


##############################
###### HELPER FUNCTIONS ######
##############################
# TODO: This and `convert_to_backend_form_like` could be merged into a single funciton
def convert_to_backend_form(x, sample_lst, img_size, backend="torch_dense"):
    if backend == "spconv":
        sparse_indices = sample_lst_to_sparse_indices(sample_lst, img_size)
        x = SparseConvTensor(
                    features=rearrange(x, "b l c -> (b l) c"),
                    indices=sparse_indices,
                    spatial_shape=(img_size, img_size),
                    batch_size=x.size(0),
                )
    elif backend == "torchsparse":
        sparse_indices = sample_lst_to_sparse_indices(sample_lst, img_size, ndims=3)
        x = torchsparse.SparseTensor(
            coords=sparse_indices,
            feats=rearrange(x, "b l c -> (b l) c")
        )
    elif backend == "minkowski":
        sparse_indices = sample_lst_to_sparse_indices(sample_lst, img_size)
        x = ME.SparseTensor(
                features=rearrange(x, "b l c -> (b l) c"),
                coordinates=sparse_indices,
                # TODO: allow this to be changed externally
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, # or MEMORY_OPTIMIZED
            )
    elif backend == "torch_geometric":
        coords = get_coords(x.size(0), img_size).to(x.device)
        coords_sampled = torch.gather(coords, 1, sample_lst.unsqueeze(2).repeat(1,1,coords.size(2))).contiguous()
        batch = get_batch(x.size(0), x.size(1)).to(sample_lst.device)
        x = TorchGeometricTensor(
            coords=rearrange(coords_sampled, "b l c -> (b l) c"),
            feats=rearrange(x, "b l c -> (b l) c"),
            batch=batch,
            )
    elif backend == "torch_dense":
        x_full = torch.zeros(x.size(0), img_size**2, x.size(2), device=x.device, dtype=x.dtype)
        x_full.scatter_(1, repeat(sample_lst, "b l -> b l c", c=x.size(-1)), x)
        x = rearrange(x_full, "b (h w) c -> b c h w", h=img_size, w=img_size)
    else:
        raise Exception("Unrecognised backend.")

    return x

def convert_to_backend_form_like(x, backend_tensor, sample_lst=None, img_size=None, backend="torch_dense", rearrange_x=True):
    if backend == "spconv":
        assert img_size is not None
        x = SparseConvTensor(
                    features=rearrange(x, "b l c -> (b l) c") if rearrange_x else x,
                    indices=backend_tensor.indices,
                    spatial_shape=(img_size, img_size),
                    batch_size=x.size(0),
                )
    elif backend == "torchsparse":
        x = torchsparse.SparseTensor(
            coords=backend_tensor.coords,
            feats=rearrange(x, "b l c -> (b l) c") if rearrange_x else x,
            stride=backend_tensor.stride
        )
        x.cmaps = backend_tensor.cmaps
        x.kmaps = backend_tensor.kmaps
    elif backend == "minkowski":
        x = ME.SparseTensor(
                features=rearrange(x, "b l c -> (b l) c") if rearrange_x else x,
                # TODO: allow this to be changed externally
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, # or MEMORY_OPTIMIZED
                coordinate_map_key=backend_tensor.coordinate_map_key,
                coordinate_manager=backend_tensor.coordinate_manager
            )
    elif backend == "torch_geometric":
        x = TorchGeometricTensor(
            feats=rearrange(x, "b l c -> (b l) c") if rearrange_x else x,
            batch=backend_tensor.batch,
            coords=backend_tensor.coords,
            edge_index=backend_tensor.edge_index,
            edge_attr=backend_tensor.edge_attr,
        )
    elif backend == "torch_dense":
        # Don't bother using the backend_tensor, just use sample_lst
        assert img_size is not None
        assert sample_lst is not None
        x_full = torch.zeros(x.size(0), img_size**2, x.size(2), device=x.device, dtype=x.dtype)
        x_full.scatter_(1, repeat(sample_lst, "b l -> b l c", c=x.size(-1)), x)
        x = rearrange(x_full, "b (h w) c -> b c h w", h=img_size, w=img_size)
    else:
        raise Exception("Unrecognised backend.")

    return x

def get_features_from_backend_form(x, sample_lst, backend="torch_dense"):
    if backend == "spconv":
        return rearrange(x.features, "(b l) c -> b l c", b=sample_lst.size(0))
    elif backend == "torchsparse":
        return rearrange(x.feats, "(b l) c -> b l c", b=sample_lst.size(0))
    elif backend == "minkowski":
        return rearrange(x.features, "(b l) c -> b l c", b=sample_lst.size(0))
    elif backend == "torch_geometric":
        return rearrange(x.feats, "(b l) c -> b l c", b=sample_lst.size(0))
    elif backend == "torch_dense":
        x = rearrange(x, "b c h w -> b (h w) c")
        x = torch.gather(x, 1, sample_lst.unsqueeze(2).repeat(1,1,x.size(2)))
        return x
    else:
        raise Exception("Unrecognised backend.")

def calculate_norm(conv, backend_tensor, sample_lst, img_size, batch_size, backend="torch_dense"):
    if backend == "spconv":
        device, dtype = backend_tensor.features.device, backend_tensor.features.dtype
        ones = torch.ones(backend_tensor.features.size(0), 1, device=device, dtype=dtype)
        mask = SparseConvTensor(
                features=ones,
                indices=backend_tensor.indices,
                spatial_shape=(img_size, img_size),
                batch_size=batch_size,
            )
        norm = conv(mask)
    elif backend == "torchsparse":
        device, dtype = backend_tensor.feats.device, backend_tensor.feats.dtype
        ones = torch.ones(backend_tensor.feats.size(0), 1, device=device, dtype=dtype)
        mask = torchsparse.SparseTensor(
                coords=backend_tensor.coords,
                feats=ones,
                stride=backend_tensor.stride
            )
        mask.cmaps = backend_tensor.cmaps
        mask.kmaps = backend_tensor.kmaps
        norm = conv(mask)
    elif backend == "minkowski":
        device, dtype = backend_tensor.features.device, backend_tensor.features.dtype
        ones = torch.ones(backend_tensor.features.size(0), 1, device=device, dtype=dtype)
        mask = ME.SparseTensor(
                features=ones,
                # TODO: allow this to be changed externally
                minkowski_algorithm=backend_tensor.coordinate_manager.minkowski_algorithm, # or MEMORY_OPTIMIZED
                coordinate_map_key=backend_tensor.coordinate_map_key,
                coordinate_manager=backend_tensor.coordinate_manager
            )
        norm = conv(mask)
    elif backend == "torch_geometric":
        return None
    elif backend == "torch_dense":
        device, dtype = backend_tensor.device, backend_tensor.dtype
        mask = torch.zeros(sample_lst.size(0), img_size**2, device=device, dtype=dtype)
        mask.scatter_(1, sample_lst, torch.ones(sample_lst.size(0), sample_lst.size(1), device=sample_lst.device, dtype=dtype))
        mask = rearrange(mask, "b (h w) -> b () h w", h=img_size, w=img_size)
        norm = conv(mask)
        norm[norm < 1e-5] = 1.0
        norm = (norm, mask)
    else:
        raise Exception("Unrecognised backend.")

    return norm

# TODO: This can be clearned up a bit
def get_normalising_conv(kernel_size, img_size=None, backend="torch_dense"):
    if backend == "spconv":
        assert SPCONV_AVAILABLE, "spconv backend is not detected."
        weight = torch.ones(1, kernel_size, kernel_size, 1) / (kernel_size ** 2)
        conv = spconv.SubMConv2d(1, 1, kernel_size=kernel_size, bias=False, padding=kernel_size//2)
        conv.weight.data = weight
        conv.weight.requires_grad_(False)
    elif backend == "torchsparse":
        assert TORCHSPARSE_AVAILABLE, "torchsparse backend is not detected."
        weight = torch.ones(kernel_size**2, 1, 1) / (kernel_size ** 2)
        conv = spnn.Conv3d(1, 1, kernel_size=(1,kernel_size,kernel_size), bias=False)
        conv.kernel.data = weight
        conv.kernel.requires_grad_(False)
    elif backend == "minkowski":
        assert MINKOWSKI_AVAILABLE, "Minkowski Engine backend is not detected."
        weight = torch.ones(kernel_size**2, 1, 1) / (kernel_size ** 2)
        conv = ME.MinkowskiConvolution(1, 1, kernel_size=kernel_size, bias=False, dimension=2)
        conv.kernel.data = weight
        conv.kernel.requires_grad_(False)
    elif backend == "torch_geometric":
        assert TORCHGEOMETRIC_AVAILABLE, "torch_geometric backend is not detected."
        conv = None
    elif backend == "torch_dense":
        weight = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=False, padding=kernel_size//2)
        conv.weight.data = weight
        conv.weight.requires_grad_(False)
        return conv
    else:
        raise Exception("Unrecognised backend.")

    return conv

"""
sample_lst is a tensor of shape (B, L)
which can be used to index flattened 2D images.
This functions converts it to a tensor of shape (BxL, 3)
    indices[:,0] is the number of the item in the batch
    indices[:,1] is the number of the item in the y direction
    indices[:,2] is the number of the item in the x direction
"""
# TODO: Any chance to get better performance by sorting in a specific way?
def sample_lst_to_sparse_indices(sample_lst, img_size, ndims=2, dtype=torch.int32):
    # number of the item in the batch - (B,)
    batch_idx = torch.arange(sample_lst.size(0), device=sample_lst.device, dtype=torch.int32)
    batch_idx = repeat(batch_idx, "b -> b l", l=sample_lst.size(1))
    # pixel number in vertical direction - (B,L)
    sample_lst_h = sample_lst.div(img_size, rounding_mode='trunc').to(dtype)
    # pixel number in horizontal direction - (B,L)
    sample_lst_w =  (sample_lst % img_size).to(dtype)

    if ndims == 2:
        indices = torch.stack([batch_idx, sample_lst_h, sample_lst_w], dim=2)
        indices = rearrange(indices, "b l three -> (b l) three")
    else:
        zeros = torch.zeros_like(sample_lst_h)
        indices = torch.stack([zeros, sample_lst_h, sample_lst_w, batch_idx], dim=2)
        indices = rearrange(indices, "b l four -> (b l) four")

    return indices

def ts_add(a, b):
    if isinstance(b, SparseTensor):
        feats = a.feats + b.feats
    else:
        feats = a.feats + b
    out = SparseTensor(
        coords=a.coords,
        feats=feats,
        stride=a.stride
    )
    out.cmaps = a.cmaps
    out.kmaps = a.kmaps
    return out

def ts_div(a, b):
    if isinstance(b, SparseTensor):
        feats = a.feats / b.feats
    else:
        feats = a.feats / b
    out = SparseTensor(
        coords=a.coords,
        feats=feats,
        stride=a.stride
    )
    out.cmaps = a.cmaps
    out.kmaps = a.kmaps
    return out

def spconv_add(a, b):
    if isinstance(b, SparseConvTensor):
        return a.replace_feature(a.features + b.features)
    else:
        return a.replace_feature(a.features + b)

def spconv_div(a, b):
    if isinstance(b, SparseConvTensor):
        return a.replace_feature(a.features / b.features)
    else:
        return a.replace_feature(a.features / b)

def spconv_clamp(a, min=None, max=None):
    return a.replace_feature(a.features.clamp(min=min, max=max))

def tg_add(a, b):
    if isinstance(b, TorchGeometricTensor):
        feats = a.feats + b.feats
    else:
        feats = a.feats + b
    out = TorchGeometricTensor(
        feats=feats,
        batch=a.batch,
        coords=a.coords,
        edge_index=a.edge_index,
        edge_attr=a.edge_attr,
    )
    return out

def tg_div(a, b):
    if isinstance(b, TorchGeometricTensor):
        feats = a.feats / b.feats
    else:
        feats = a.feats / b
    out = TorchGeometricTensor(
        feats=feats,
        batch=a.batch,
        coords=a.coords,
        edge_index=a.edge_index,
        edge_attr=a.edge_attr,
    )
    return out

class MinkowskiLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, input):
        output = self.norm(input.F)
        if isinstance(input, ME.TensorField):
            return ME.TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return ME.SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )

def minkowski_clamp(x, min=None, max=None):
    output = x.features.clamp(min=min, max=max)
    if isinstance(x, ME.TensorField):
        return ME.TensorField(
            output,
            coordinate_field_map_key=x.coordinate_field_map_key,
            coordinate_manager=x.coordinate_manager,
            quantization_mode=x.quantization_mode,
        )
    else:
        return ME.SparseTensor(
            output,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )

class ImageLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import numpy as np
# from flash_attn.flash_attention import FlashAttention


class UNO(nn.Module):
    def __init__(self, nin, nout, width=64, mults=(1,2,4,8), blocks_per_level=(2,2,2,2), time_emb_dim=None,
                 z_dim=None, conv_type="conv", res=64, attn_res=[16,8], dropout_res=16, dropout=0.1):
        super().__init__()
        self.width = width
        self.conv_type = conv_type
        self.fc0 = nn.Conv2d(nin+2 if conv_type == "spectral" else nin, width, 1)

        dims = [width, *map(lambda m: width * m, mults)]
        in_out = list(zip(dims[:-1], dims[1:], blocks_per_level))
        cur_res = res

        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out, num_blocks) in enumerate(in_out):
            is_last = ind == len(mults) - 1
            cur_dropout = dropout if cur_res <= dropout_res else 0.0
            layers = nn.ModuleList([])
            for _ in range(num_blocks):
                layers.append(nn.ModuleList([
                    ConvBlock(dim_in, dim_in, emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, dropout=cur_dropout),
                    FlashAttnBlock(dim_in, emb_dim=time_emb_dim, z_dim=z_dim) if cur_res in attn_res else Identity(),
                ]))

            downsample = get_conv(conv_type, dim_in, dim_out, downsample=True) if not is_last else get_conv(conv_type, dim_in, dim_out)
            self.downs.append(nn.ModuleList([layers, downsample]))
            cur_res = cur_res // 2 if not is_last else cur_res

        self.mid_block1 = ConvBlock(dim_out, dim_out, emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, dropout=cur_dropout)
        self.mid_attn = FlashAttnBlock(dim_out, emb_dim=time_emb_dim, z_dim=z_dim)
        self.mid_block2 = ConvBlock(dim_out, dim_out, emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, dropout=cur_dropout)

        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out, num_blocks) in enumerate(reversed(in_out)):
            is_last = ind == len(mults) - 1
            cur_dropout = dropout if cur_res <= dropout_res else 0.0
            layers = nn.ModuleList([])
            for _ in range(num_blocks):
                layers.append(nn.ModuleList([
                    ConvBlock(dim_out + dim_in, dim_out, emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, dropout=cur_dropout),
                    FlashAttnBlock(dim_out, emb_dim=time_emb_dim, z_dim=z_dim) if cur_res in attn_res else Identity(),
                ]))

            upsample = get_conv(conv_type, dim_out, dim_in, upsample=True) if not is_last else get_conv(conv_type, dim_out, dim_in)
            self.ups.append(nn.ModuleList([layers, upsample]))
            cur_res = cur_res * 2 if not is_last else cur_res

        self.fc1 = nn.Conv2d(width, 128, 1)
        self.fc2 = nn.Conv2d(128, nout, 1)

    def forward(self, x, emb=None, z=None):
        # NOTE: Get grid can probably be replaced with fourier features or something?
        if self.conv_type == "spectral":
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=1)
        x = self.fc0(x)

        h = []
        for level, down in self.downs:
            for layers in level:
                for layer in layers:
                    x = layer(x, emb, z=z)
                h.append(x)
            x = down(x)

        x = self.mid_block1(x, emb, z=z)
        x = self.mid_attn(x, emb, z=z)
        x = self.mid_block2(x, emb, z=z)

        for level, up in self.ups:
            for layers in level:
                h_pop = h.pop()
                x = torch.cat((x, h_pop), dim=1)
                for layer in layers:
                    x = layer(x, emb, z=z)
            x = up(x)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x

    def get_grid(self, shape, device):
        batchsize, _, size_x, size_y = shape
        gridx = torch.tensor(np.linspace(0, size_x-1, size_x)/size_x, dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, size_y-1, size_y)/size_y, dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

class UNOEncoder(nn.Module):
    def __init__(self, nin, nout, width=64, mults=(1,2,4,8), blocks_per_level=(2,2,2,2), time_emb_dim=None,
                 z_dim=None, conv_type="conv", res=64, attn_res=[16,8], dropout_res=16, dropout=0.1):
        super().__init__()
        self.width = width
        self.conv_type = conv_type
        self.fc0 = nn.Conv2d(nin+2 if conv_type == "spectral" else nin, width, 1)

        dims = [width, *map(lambda m: width * m, mults)]
        in_out = list(zip(dims[:-1], dims[1:], blocks_per_level))
        cur_res = res

        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out, num_blocks) in enumerate(in_out):
            is_last = ind == len(mults) - 1
            cur_dropout = dropout if cur_res <= dropout_res else 0.0
            layers = nn.ModuleList([])
            for _ in range(num_blocks):
                layers.append(nn.ModuleList([
                    ConvBlock(dim_in, dim_in, emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, dropout=cur_dropout),
                    FlashAttnBlock(dim_in, emb_dim=time_emb_dim, z_dim=z_dim) if cur_res in attn_res else Identity(),
                ]))

            downsample = get_conv(conv_type, dim_in, dim_out, downsample=True) if not is_last else get_conv(conv_type, dim_in, dim_out)
            self.downs.append(nn.ModuleList([layers, downsample]))
            cur_res = cur_res // 2 if not is_last else cur_res

        self.mid_block1 = ConvBlock(dim_out, dim_out, emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, dropout=cur_dropout)
        self.mid_attn = FlashAttnBlock(dim_out, emb_dim=time_emb_dim, z_dim=z_dim)
        self.mid_block2 = ConvBlock(dim_out, dim_out, emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, dropout=cur_dropout)

        self.fc1 = nn.Conv2d(dim_out, nout, 1)

    def forward(self, x, emb=None, z=None):
        # NOTE: Get grid can probably be replaced with fourier features or something?
        if self.conv_type == "spectral":
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=1)
        x = self.fc0(x)

        for level, down in self.downs:
            for layers in level:
                for layer in layers:
                    x = layer(x, emb, z=z)
            x = down(x)

        x = self.mid_block1(x, emb, z=z)
        x = self.mid_attn(x, emb, z=z)
        x = self.mid_block2(x, emb, z=z)

        x = self.fc1(x)

        return x

    def get_grid(self, shape, device):
        batchsize, _, size_x, size_y = shape
        gridx = torch.tensor(np.linspace(0, size_x-1, size_x)/size_x, dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, size_y-1, size_y)/size_y, dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

def get_conv(conv_type, nin, nout, downsample=False, upsample=False):
    if conv_type == "conv":
        if downsample:
            return nn.Conv2d(nin, nout, 3, stride=2, padding=1)
        elif upsample:
            return nn.ConvTranspose2d(nin, nout, 4, stride=2, padding=1)
        else:
            return nn.Conv2d(nin, nout, 3, 1, 1)
    elif conv_type == "spectral":
        # Same as conv but the kernel is defined in fourier space,
        # allowing better generalisation to different resolutions.
        # Does not support AMP.
        # TODO: 4 modes or 3???
        if downsample:
            return SpectralConv2d(nin, nout, 4, 4, out_mult=0.5)
        elif upsample:
            return SpectralConv2d(nin, nout, 4, 4, out_mult=2.0)
        else:
            return SpectralConv2d(nin, nout, 4, 4)
    else:
        raise Exception("Unknown Convolution name. Expected either 'conv' or 'spectral'")


class ConvBlock(nn.Module):
    def __init__(self, nin, nout, emb_dim=None, z_dim=None, dropout=0.0, conv_type="conv"):
        super().__init__()
        self.in_layers = nn.Sequential(
            LayerNorm((1, nin, 1, 1)), # TODO: Use GroupNorm instead like BeatGANs? Spatial though
            nn.GELU(), # TODO: Use SiLU instead like BeatGANs?
            get_conv(conv_type, nin, nout),
        )
        self.norm = LayerNorm((1, nout, 1, 1))
        self.out_layers = nn.Sequential(
            nn.GELU(),
            nn.Dropout(p=dropout),
            get_conv(conv_type, nout, nout)
        )
        self.res_conv = nn.Conv2d(nin, nout, 1) if nin != nout else nn.Identity()

        if emb_dim is not None:
            self.time = nn.Sequential(nn.GELU(), nn.Linear(emb_dim, nout*2))
        if z_dim is not None:
            self.z_mlp = nn.Sequential(nn.Linear(z_dim, nout), nn.GELU(), nn.Linear(nout, nout))

    def forward(self, x, t=None, z=None):
        # TODO: Should be blocks be made more like FNO blocks???
        h = self.in_layers(x)

        # Condition on t and z
        h = self.norm(h)
        if t is not None:
            t_scale, t_shift = self.time(t)[:,:,None,None].chunk(2, dim=1)
            h = h * (1 + t_scale) + t_shift
        if z is not None:
            z_scale = self.z_mlp(z)[:,:,None,None]
            h = h * (1 + z_scale)

        h = self.out_layers(h)

        return h + self.res_conv(x)

# class GroupNorm(nn.GroupNorm):
#     def forward(self, x):
#         return super().forward(x.float()).type(x.dtype)

# class FlashAttnBlock(nn.Module):
#     def __init__(self, dim, min_heads=4, dim_head=32, mult=2, emb_dim=None, z_dim=None):
#         super().__init__()
#         self.num_heads = num_heads = max(dim // dim_head, min_heads)
#         self.norm = nn.LayerNorm(dim)
#         self.qkv = nn.Linear(dim, num_heads*dim_head*3)
#         self.attn_linear = nn.Linear(num_heads*dim_head, dim)
#         self.attn = FlashAttention()

#     def forward(self, x, t=None, z=None):
#         height = x.size(2)
#         h = rearrange(x, "b c h w -> b (h w) c")
#         qkv = self.qkv(self.norm(h))
#         # split qkv and separate heads
#         qkv = rearrange(qkv, "b l (three h c) -> b l three h c", three=3, h=self.num_heads)

#         # Do Flash Attention
#         h, _ = self.attn(qkv)

#         h = rearrange(h, "b l h c -> b l (h c)")
#         h = self.attn_linear(h)
#         h = rearrange(h, "b (h w) c -> b c h w", h=height)

#         return x + h

class FlashAttnBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, t=None, z=None):
        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

class LayerNorm(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.weight + self.bias

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, out_shape=None, out_mult=None):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        assert not (out_shape is not None and out_mult is not None), "Both out_shape or out_mult can't be set at once"
        self.out_shape, self.out_mult = None, None
        if out_shape is not None:
            self.out_shape = out_shape if isinstance(out_shape, tuple) else (out_shape, out_shape)
        if out_mult is not None:
            self.out_mult = out_mult if isinstance(out_mult, tuple) else (out_mult, out_mult)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input.to(weights.dtype), weights).to(input.dtype) # or input to float32?

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x) # TODO: Make this norm="forward"?

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        if self.out_shape is not None:
            # change shape to self.out_shape
            x = torch.fft.irfft2(out_ft, s=self.out_shape)
        elif self.out_mult:
            # change shape to multiple of current shape
            out_shape = (int(x.size(-2) * self.out_mult[0]), int(x.size(-1) * self.out_mult[1]))
            x = torch.fft.irfft2(out_ft, s=out_shape)
        else:
            # keep shape the same
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch3d.ops import knn_points, knn_gather
import math
import warnings

# from .conv_uno import UNO, UNOEncoder
# from .sparse_conv_block import SparseConvResBlock
# from .sparse_conv_block import convert_to_backend_form, convert_to_backend_form_like, \
#     calculate_norm, get_features_from_backend_form, get_normalising_conv

class SparseUNet(nn.Module):
    def __init__(self, channels=3, nf=64, time_emb_dim=256, img_size=128, num_conv_blocks=3, knn_neighbours=3, uno_res=64,
                 uno_mults=(1,2,4,8), z_dim=None, out_channels=None, conv_type="conv",
                 depthwise_sparse=True, kernel_size=7, backend="torch_dense", optimise_dense=True,
                 blocks_per_level=(2,2,2,2), attn_res=[16,8], dropout_res=16, dropout=0.1,
                 uno_base_nf=64, fc_channels=32,
                 ):
        super().__init__()
        self.backend = backend
        self.img_size = img_size
        self.uno_res = uno_res
        self.knn_neighbours = knn_neighbours
        self.kernel_size = kernel_size
        self.optimise_dense = optimise_dense
        # Input projection
        self.linear_in = nn.Linear(channels, nf)
        # Output projection
        self.linear_out = nn.Linear(nf, out_channels if out_channels is not None else channels)

        # Diffusion time MLP
        # TODO: Better to have more features here? 64 by default isn't many
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        uno_coords = torch.stack(torch.meshgrid(*[torch.linspace(0, uno_res-1, steps=uno_res)/uno_res for _ in range(2)], indexing='ij'))
        uno_coords = rearrange(uno_coords, 'c h w -> () (h w) c')
        self.register_buffer("uno_coords", uno_coords)

        self.normalising_conv = get_normalising_conv(kernel_size=kernel_size, backend=backend)

        self.down_blocks = nn.ModuleList([])
        for _ in range(num_conv_blocks):
            self.down_blocks.append(SparseConvResBlock(
                img_size, nf, kernel_size=kernel_size, mult=2, time_emb_dim=time_emb_dim, z_dim=z_dim, backend=backend, depthwise=depthwise_sparse, fc_channels=fc_channels,
            ))
        self.uno_linear_in = nn.Linear(nf, uno_base_nf)

        self.uno_linear_out = nn.Linear(uno_base_nf, nf)
        self.up_blocks = nn.ModuleList([])
        for _ in range(num_conv_blocks):
            self.up_blocks.append(SparseConvResBlock(
                img_size, nf, kernel_size=kernel_size, mult=2, skip_dim=nf, time_emb_dim=time_emb_dim, z_dim=z_dim, backend=backend, depthwise=depthwise_sparse, fc_channels=fc_channels,
            ))

        self.uno = UNO(uno_base_nf, uno_base_nf, width=uno_base_nf, mults=uno_mults, blocks_per_level=blocks_per_level,
                       time_emb_dim=time_emb_dim, z_dim=z_dim, conv_type=conv_type, res=uno_res,
                       attn_res=attn_res, dropout_res=dropout_res, dropout=dropout)

    def knn_interpolate_to_grid(self, x, coords):
        with torch.no_grad():
            _, assign_index, neighbour_coords = knn_points(self.uno_coords.repeat(x.size(0),1,1), coords, K=self.knn_neighbours, return_nn=True)
            # neighbour_coords: (B, y_length, K, 2)
            diff = neighbour_coords - self.uno_coords.unsqueeze(2) # can probably use dist from knn_points
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(squared_distance, min=1e-16) # (B, y_length, K, 1)

        # See Eqn. 2 in PointNet++. Inverse square distance weighted mean
        neighbours = knn_gather(x, assign_index) # (B, y_length, K, C)
        out = (neighbours * weights).sum(2) / weights.sum(2)

        return out.to(x.dtype)

    def get_torch_norm_kernel_size(self, img_size, round_down=True):
        if img_size != self.img_size:
            ratio = img_size / self.img_size
            # new kernel_size becomes:
            # 1 -> 1, 1.5 -> 1, 2 -> 1 or 3, 2.5 -> 3, 3 -> 3, 3.5 -> 3, 4 -> 3 or 5, 4.5 -> 5, ...
            # where there are multiple options this is determined by round_down
            new_kernel_size = self.kernel_size * ratio
            if round_down:
                new_kernel_size = 2 * round((new_kernel_size - 1) / 2) + 1
            else:
                new_kernel_size = math.floor(new_kernel_size / 2) * 2 + 1
            return max(new_kernel_size, 3)
        else:
            return self.kernel_size

    def dense_forward(self, x, t, z=None):
        # If x is image shaped (4D) then treat it as a dense tensor for better optimisation
        height = x.size(2)

        coords = torch.stack(torch.meshgrid(*[torch.linspace(0, height-1, steps=height)/height for _ in range(2)], indexing='ij')).to(x.device)
        coords = rearrange(coords, 'c h w -> () (h w) c')
        coords = repeat(coords, "() ... -> b ...", b=x.size(0))

        x = F.conv2d(x, self.linear_in.weight[:,:,None,None], bias=self.linear_in.bias)
        t = self.time_mlp(t)

        # NOTE: Still need to norm to avoid edge artefacts
        mask = torch.ones(x.size(0), 1, x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        kernel_size = self.get_torch_norm_kernel_size(height)
        weight = torch.ones(1, 1, kernel_size, kernel_size, dtype=x.dtype, device=x.device) / (self.kernel_size ** 2)
        norm = F.conv2d(mask, weight, padding=kernel_size//2)

        # 1. Down conv blocks
        downs = []
        for block in self.down_blocks:
            x = block(x, t=t, z=z, norm=norm)
            downs.append(x)

        # 2. Interpolate to regular grid
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.uno_linear_in(x)
        x = self.knn_interpolate_to_grid(x, coords)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.uno_res)

        # 3. UNO
        x = self.uno(x, t, z=z)

        # 4. Interpolate back to sparse coordinates
        x = F.grid_sample(x, coords.unsqueeze(2), mode='bilinear', align_corners=True)
        x = rearrange(x, "b c (h w) () -> b c h w", h=height)
        x = F.conv2d(x, self.uno_linear_out.weight[:,:,None,None], bias=self.uno_linear_out.bias)

        # 5. Up conv blocks
        for block in self.up_blocks:
            skip = downs.pop()
            x = block(x, t=t, z=z, skip=skip, norm=norm)

        x = F.conv2d(x, self.linear_out.weight[:,:,None,None], bias=self.linear_out.bias)

        return x

    def forward(self, x, t, z=None, sample_lst=None, coords=None, img_size=None):
        batch_size = x.size(0)

        # If x is image shaped (4D) then treat it as a dense tensor for better optimisation
        if len(x.shape) == 4 and self.optimise_dense:
            if sample_lst is not None:
                warnings.warn("Ignoring sample_lst: Recieved 4D x and sample_list != None so treating x as a dense Image.")
            if coords is not None:
                warnings.warn("Ignoring coords: Recieved 4D x and coords != None so treating x as a dense Image.")
            return self.dense_forward(x, t, z=z)

        # TODO: Re-add the parts of this needed if x is image shape but optimise_dense is False
        # i.e. rearrange and set sample_lst.

        assert sample_lst is not None, "In sparse mode sample_lst must be provided"
        img_size = img_size if img_size is not None else self.img_size
        if coords is None:
            coords = torch.stack(torch.meshgrid(*[torch.linspace(0, img_size-1, steps=img_size)/img_size for _ in range(2)], indexing='ij')).to(x.device)
            coords = rearrange(coords, 'c h w -> () (h w) c')
            coords = repeat(coords, "() ... -> b ...", b=x.size(0))
            coords = torch.gather(coords, 1, sample_lst.unsqueeze(2).repeat(1,1,coords.size(2))).contiguous()

        x = self.linear_in(x)
        t = self.time_mlp(t)

        # 1. Down conv blocks
        # Cache mask and norms
        x = convert_to_backend_form(x, sample_lst, img_size, backend=self.backend)
        backend_tensor = x
        norm = calculate_norm(self.normalising_conv, backend_tensor, sample_lst, img_size, batch_size, backend=self.backend)

        downs = []
        for block in self.down_blocks:
            x = block(x, t=t, z=z, norm=norm)
            downs.append(x)

        # 2. Interpolate to regular grid
        x = get_features_from_backend_form(x, sample_lst, backend=self.backend)
        x = self.uno_linear_in(x)
        x = self.knn_interpolate_to_grid(x, coords)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.uno_res)

        # 3. UNO
        x = self.uno(x, t, z=z)

        # 4. Interpolate back to sparse coordinates
        x = F.grid_sample(x, coords.unsqueeze(2), mode='bilinear', align_corners=True)
        x = rearrange(x, "b c l () -> b l c")
        x = self.uno_linear_out(x)
        x = convert_to_backend_form_like(x, backend_tensor, sample_lst=sample_lst, img_size=img_size, backend=self.backend)

        # 5. Up conv blocks
        for block in self.up_blocks:
            skip = downs.pop()
            x = block(x, t=t, z=z, skip=skip, norm=norm)

        x = get_features_from_backend_form(x, sample_lst, backend=self.backend)
        x = self.linear_out(x)

        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
