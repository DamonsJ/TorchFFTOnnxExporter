import torch
from torch import Tensor
from torch.fft import fftn, fftshift, ifftn, ifftshift
from typing import Sequence

# 定义公共接口
__all__ = ["real_fftn","real_fftshift","real_ifftshift","real_ifftn"]
#  返回一个CustomOpDef对象
@torch.library.custom_op("AIP::real_fftn", mutates_args=())
def real_fftn(x:Tensor,dim: Sequence[int])-> Tensor:
    x_freq = fftn(x, dim = dim)
    return torch.view_as_real(x_freq)

@torch.library.custom_op("AIP::real_fftshift", mutates_args=())
def real_fftshift(x:Tensor,dim: Sequence[int])-> Tensor:
    x = torch.view_as_complex(x)
    x_freq = fftshift(x, dim = dim)
    return torch.view_as_real(x_freq)

@torch.library.custom_op("AIP::real_ifftshift", mutates_args=())
def real_ifftshift(x:Tensor, mask:Tensor, dim: Sequence[int])-> Tensor:
    x = torch.view_as_complex(x)
    x = x * mask
    x_freq = ifftshift(x, dim = dim)
    return torch.view_as_real(x_freq)

@torch.library.custom_op("AIP::real_ifftn", mutates_args=())
def real_ifftn(x:Tensor,dim: Sequence[int])-> Tensor:
    x = torch.view_as_complex(x)
    x_freq = ifftn(x, dim = dim)
    return x_freq.real


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("AIP::real_fftn")
def real_fftn_abstract(a, dim):
    complex_shape = list(a.shape)
    complex_shape = complex_shape + [2]
    return torch.zeros(complex_shape)

@torch.library.register_fake("AIP::real_fftshift")
def real_fftshift_abstract(a, dim):
    return torch.empty_like(a)

@torch.library.register_fake("AIP::real_ifftshift")
def real_ifftshift_abstract(a, mask,dim):
    return torch.empty_like(a)

@torch.library.register_fake("AIP::real_ifftn")
def real_ifftn_abstract(a, dim):
    complex_shape = list(a.shape)
    complex_shape = complex_shape[:-1]
    return torch.zeros(complex_shape)