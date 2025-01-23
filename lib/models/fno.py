# from lib.models.fourier_unet_modified import FNOUNet2d
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, rand=True, separable=False):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))

        self.separable = separable
        if self.separable:
            if rand:
                self._weights1a = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 1, dtype=torch.cfloat))
                self._weights1b = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 1, self.modes2, dtype=torch.cfloat))
                self._weights2a = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 1, dtype=torch.cfloat))
                self._weights2b = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 1, self.modes2, dtype=torch.cfloat))
            else:
                self._weights1a = nn.Parameter(0.2+0.8*torch.rand(in_channels, out_channels, self.modes1, 1, dtype=torch.cfloat))
                self._weights1b = nn.Parameter(0.2+0.8*torch.rand(in_channels, out_channels, 1, self.modes2, dtype=torch.cfloat))
                self._weights2a = nn.Parameter(0.2+0.8*torch.rand(in_channels, out_channels, self.modes1, 1, dtype=torch.cfloat))
                self._weights2b = nn.Parameter(0.2+0.8*torch.rand(in_channels, out_channels, 1, self.modes2, dtype=torch.cfloat))
        else:
            if rand:
                self._weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
                self._weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
            else:
                self._weights1 = nn.Parameter(0.2+0.8*torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
                self._weights2 = nn.Parameter(0.2+0.8*torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    @property
    def weights1(self):
        if self.separable:
            return self._weights1a * self._weights1b
        else:
            return self._weights1

    @property
    def weights2(self):
        if self.separable:
            return self._weights2a * self._weights2b
        else:
            return self._weights2

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, dim=(-2,-1))

        # Multiply relevant Fourier modes #oneside
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def extra_repr(self) -> str:
        return f'separable={self.separable}'


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_ch=1, out_ch=None, separable=False):
        super(FNO2d, self).__init__()
        if out_ch is None:
            out_ch = in_ch

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, separable=separable)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, separable=separable)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, separable=separable)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, separable=separable)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, separable=separable)
        self.conv6 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, separable=separable)
        self.conv5 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, separable=separable)


        self.w0 = nn.Conv2d(self.width, self.width, 1, padding = 'same')
        self.w1 = nn.Conv2d(self.width, self.width, 1, padding = 'same')
        self.w2 = nn.Conv2d(self.width, self.width, 1, padding = 'same')
        self.w3 = nn.Conv2d(self.width, self.width, 1, padding = 'same')
        self.w4 = nn.Conv2d(self.width, self.width, 1, padding = 'same')
        self.w5 = nn.Conv2d(self.width, self.width, 1, padding = 'same')
        self.w6 = nn.Conv2d(self.width, self.width, 1, padding = 'same')

        self.fc0 = nn.Linear(3 + in_ch, self.width) # input channel is 3: (a(x, y), x, y, t)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_ch)


    def forward(self, x, t=None):
        if t is None:
            t = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        t = t[:,None,None,None].repeat(1,1,x.shape[2],x.shape[3])
        t = t.permute(0,2,3,1)
        x = x.permute(0,2,3,1)
        grid = self.get_grid(x.shape, x.device)

        x = torch.cat((x, grid, t), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv4(x)
        x2 = self.w4(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv5(x)
        x2 = self.w5(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv6(x)
        x2 = self.w6(x)
        x = x1 + x2
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x).permute(0,3,1,2)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

