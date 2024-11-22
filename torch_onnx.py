import torch
from torch.fft import fftn, fftshift, ifftn, ifftshift
import onnx
from onnx import helper, TensorProto
from torch.onnx import symbolic_helper
from torch.onnx.symbolic_helper import parse_args
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import _get_tensor_dim_size, _get_tensor_sizes
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import aip_fft

@parse_args("v", "v")
def custom_fft_fftn_symbolic(g, s, dim):
    """
    Define the symbolic function for fft_fftn.

    Args:
        g (torch._C.Graph): ONNX graph.
        s (torch._C.Value): Input tensor.
        dim (torch._C.Value): Dimensions to transform.

    Returns:
        torch._C.Value: ONNX node representing fft_fftn.
    """
    # Convert optional lists to tensors if needed
    dim_t = symbolic_helper._maybe_get_scalar(dim)
    input_shape = _get_tensor_sizes(s)
    if input_shape is None:
        raise RuntimeError("Shape inference failed: input shape is unknown.")
    output_shape = list(input_shape)
    output_shape.append(2)
    # Define the output type as torch.complex64

    # Create a custom ONNX operation
    output =  g.op(
        "AIP::fft_fftn",  # Custom operator name
        s,
        dim_t,
        outputs=1
    )
    output.setType(s.type().with_sizes(output_shape))
    return output

@parse_args("v", "v")
def custom_fft_fftshift_symbolic(g, s, dim):
    # Convert optional lists to tensors if needed
    dim_t = symbolic_helper._maybe_get_scalar(dim)
    # Create a custom ONNX operation
    return g.op(
        "AIP::fft_fftshift",  # Custom operator name
        s,
        dim_t,
        outputs=1
    ).setType(s.type())

@parse_args("v", "v", "v")
def custom_fft_ifftshift_symbolic(g, s, m, dim):
    # Convert optional lists to tensors if needed
    dim_t = symbolic_helper._maybe_get_scalar(dim)
    # Create a custom ONNX operation
    return g.op(
        "AIP::fft_ifftshift",  # Custom operator name
        s,
        m,
        dim_t,
        outputs=1
    ).setType(s.type())

@parse_args("v", "v")
def custom_fft_ifftn_symbolic(g, s, dim):
    # Convert optional lists to tensors if needed
    dim_t = symbolic_helper._maybe_get_scalar(dim)
    input_shape = _get_tensor_sizes(s)
    if input_shape is None:
        raise RuntimeError("Shape inference failed: input shape is unknown.")
    output_shape = list(input_shape)
    output_shape = output_shape[:-1]
    # Create a custom ONNX operation
    return g.op(
        "AIP::fft_ifftn",  # Custom operator name
        s,
        dim_t,
        outputs=1
    ).setType(s.type().with_sizes(output_shape))

class FFTONNXModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = x_in
        B, C, H, W = x.shape
        threshold = 10
        scale = 10
        # Non-power of 2 images must be float32
        if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
            x = x.to(dtype=torch.float32)

        # FFT
        x_freq = aip_fft.aip_fft.real_fftn(x, dim=(-2, -1))
        x_freq = aip_fft.aip_fft.real_fftshift(x_freq, dim=(-2, -1))
        print(f"after real_fftshift shape : {x_freq.shape}")
        B, C, H, W,_ = x_freq.shape
        mask = torch.ones((B, C, H, W), device=x.device)

        crow, ccol = H // 2, W // 2
        mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale
        # x_freq = x_freq * mask

        # IFFT
        x_freq = aip_fft.aip_fft.real_ifftshift(x_freq, mask,dim=(-2, -1))
        print(f"after real_ifftshift shape : {x_freq.shape}")
        x_filtered = aip_fft.aip_fft.real_ifftn(x_freq, dim=(-2, -1))
        print(f"after real_ifftn shape : {x_filtered.shape}")

        return x_filtered.to(dtype=x_in.dtype)

def print_onnx_graph(onnx_path):
    # 加载导出的 ONNX 模型
    model = onnx.load(onnx_path)

    # 打印模型信息
    print(onnx.helper.printable_graph(model.graph))

if __name__ == "__main__":
    export_onnx = True
    model = FFTONNXModel()
    model.eval()
    sample_input = torch.randn((2,640,96,96)).to(device="cuda", dtype=torch.float16)

    if export_onnx:
        register_custom_op_symbolic("AIP::real_fftn", custom_fft_fftn_symbolic, 17)
        register_custom_op_symbolic("AIP::real_ifftn", custom_fft_ifftn_symbolic, 17)
        register_custom_op_symbolic("AIP::real_fftshift", custom_fft_fftshift_symbolic, 17)
        register_custom_op_symbolic("AIP::real_ifftshift", custom_fft_ifftshift_symbolic, 17)

        input_names = ["x_in"]
        output_names = ["out"]
        dynamic_axes = {
                'x_in': {0: "B"},
                'out':{0: "B"},
            }
        torch.onnx.export(
                        model,
                        sample_input,
                        "fft.onnx",
                        export_params=True,
                        opset_version=17,
                        do_constant_folding=True,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                    )

        print_onnx_graph("fft.onnx")
    else:
        model(sample_input)