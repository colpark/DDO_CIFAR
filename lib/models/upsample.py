'''
copied and modified from https://github.com/PaulLyonel/multilevelDiff/blob/main/utils.py#L9
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def downsampling_fourier(input_tensor, scaling_factor:int=2):
    _ , _, height, width = input_tensor.shape  #B x C x H x W

    # select center pixels
    center_x = height//2
    center_y = width//2

    crop_dim_x = int(center_x//scaling_factor)
    crop_dim_y = int(center_y//scaling_factor)

    fimage = torch.fft.fftshift(torch.fft.fft2(input_tensor, norm="forward"))

    fft_crop = fimage[:,:,(center_x-crop_dim_x):(center_x+crop_dim_x),(center_y-crop_dim_y):(center_y+crop_dim_y)]

    tensor_downsampled = torch.real(torch.fft.ifft2(torch.fft.ifftshift(fft_crop), norm="forward"))

    return tensor_downsampled

def upsampling_fourier(input_tensor, scaling_factor:int=2):
    batch_size, channel, height, width = input_tensor.shape  #B x C x H x W

    # new height
    up_height = height*scaling_factor
    up_width = width*scaling_factor

    # select center pixels
    center_x = up_height//2
    center_y = up_width//2

    crop_dim_x = int(center_x//scaling_factor)
    crop_dim_y = int(center_y//scaling_factor)

    fimage = torch.fft.fftshift(torch.fft.fft2(input_tensor, norm="forward"))

    fft_up = torch.zeros(batch_size, channel, up_height, up_width, dtype=fimage.dtype, device=fimage.device)
    fft_up[:,:,(center_x-crop_dim_x):(center_x+crop_dim_x),(center_y-crop_dim_y):(center_y+crop_dim_y)] = fimage

    tensor_upsampled = torch.real(torch.fft.ifft2(torch.fft.ifftshift(fft_up), norm="forward"))

    return tensor_upsampled


class Upsample2d(nn.Module):
    def __init__(self, scale_factor:int=2, padding:bool=False, mode:str='replicate'):
        super().__init__()
        assert np.exp2(int(np.log2(scale_factor))) == scale_factor, 'scale_factor is a power of two.'
        assert mode in ['reflect', 'circular', 'replicate']
        self.scale_factor = scale_factor
        self.padding = padding
        self.mode = mode

    def forward(self, x, **kwargs):
        if self.padding:
            _, _, height, width = x.shape
            pad_h = height//2
            pad_w = width//2
            x_padded = F.pad(x, (pad_h, pad_h, pad_w, pad_w), mode=self.mode)
            # x_padded = pad2d_reflect(x, (pad_h, pad_h, pad_w, pad_w))
            x_up = upsampling_fourier(x_padded, self.scale_factor)
            return x_up[:,:,self.scale_factor*pad_h:-self.scale_factor*pad_h,self.scale_factor*pad_w:-self.scale_factor*pad_w]
        else:
            return upsampling_fourier(x, self.scale_factor)

    def extra_repr(self):
        return '\n'.join([
            f'scale_factor={self.scale_factor:g},',
            f'padding={self.padding:g},',
        ])


class Downsample2d(nn.Module):
    def __init__(self, scale_factor:int=2, padding:bool=False, mode:str='replicate'):
        super().__init__()
        assert np.exp2(int(np.log2(scale_factor))) == scale_factor, 'scale_factor is a power of two.'
        assert mode in ['reflect', 'circular', 'replicate']
        self.scale_factor = scale_factor
        self.padding = padding
        self.mode = mode

    def forward(self, x, **kwargs):
        if self.padding:
            _, _, height, width = x.shape
            pad_h = height//2
            pad_w = width//2
            x_padded = F.pad(x, (pad_h, pad_h, pad_w, pad_w), mode=self.mode)
            # x_padded = pad2d_reflect(x, (pad_h, pad_h, pad_w, pad_w))
            x_down = downsampling_fourier(x_padded, self.scale_factor)
            unpad_h = pad_h // self.scale_factor
            unpad_w = pad_w // self.scale_factor
            return x_down[:,:,unpad_h:-unpad_h,unpad_w:-unpad_w]
        else:
            return downsampling_fourier(x, self.scale_factor)

    def extra_repr(self):
        return '\n'.join([
            f'scale_factor={self.scale_factor:g},',
            f'padding={self.padding:g},',
        ])


class Downsample2dv2(nn.Module):
    def __init__(self, scale_factor:int=2):
        super().__init__()
        assert np.exp2(int(np.log2(scale_factor))) == scale_factor, 'scale_factor is a power of two.'
        self.scale_factor = scale_factor

    def forward(self, x, **kwargs):
        _, _, height, width = x.shape
        pad_h = height//2
        pad_w = width//2
        x_padded = F.pad(x, (pad_h, pad_h, pad_w, pad_w), mode='reflect')
        x_down = downsampling_fourier(x_padded, self.scale_factor)
        unpad_h = pad_h // self.scale_factor
        unpad_w = pad_w // self.scale_factor
        return x_down[:,:,unpad_h:-unpad_h,unpad_w:-unpad_w]

    def extra_repr(self):
        return '\n'.join([f'scale_factor={self.scale_factor:g},',])


