import os
import torch
from torch.utils.cpp_extension import load

# Load and compile on-the-fly
current_dir = os.path.dirname(os.path.abspath(__file__))
quantizer_cpp = load(
  name="quantizer",
  sources=[
    os.path.join(current_dir, "quantizer.cpp"),
    os.path.join(current_dir, "src/quant.cpp")
  ],
  extra_include_paths=[os.path.join(current_dir, "src")],
  verbose=True
)

def quantize_q2_k(tensor: torch.Tensor) -> torch.Tensor:
  """
  Quantize a 2D float32 tensor to Q2_K format.
  
  Args:
    tensor: Input tensor of shape (M, N) where N must be a multiple of 256
  
  Returns:
    Quantized tensor of type uint8 containing the block_q2_K data
  """
  if tensor.dim() != 2:
    raise ValueError("Input tensor must be 2-dimensional")
  if tensor.dtype != torch.float32:
    raise ValueError("Input tensor must be float32")
  if tensor.shape[1] % 256 != 0:
    raise ValueError("Input tensor's second dimension must be a multiple of 256")
  if not tensor.is_contiguous():
    tensor = tensor.contiguous()
      
  return quantizer_cpp.quantize_q2_k(tensor) 