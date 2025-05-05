import torch
import quantizer_cpp
from typing import Literal

def k_quantize(tensor: torch.Tensor, method: Literal["q2_k", "q3_k"]) -> torch.Tensor:
  """
  Quantize a 2D float32 tensor to Q2_K or Q3_K format.
  
  Args:
    tensor: Input tensor of shape (M, N) where N must be a multiple of 256
  
  Returns:
    Quantized tensor of type uint8 and shape (M, sizeof(block_q2_K) * N/256) containing the block_q2_K data
  """ 
  if method == "q2_k":
    return quantizer_cpp.quantize_q2_k(tensor) 
  elif method == "q3_k":
    return quantizer_cpp.quantize_q3_k(tensor) 
  else:
    raise ValueError(f"Invalid method: {method}")
