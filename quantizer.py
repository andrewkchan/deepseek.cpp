import torch
import quantizer_cpp

def quantize_q2_k(tensor: torch.Tensor) -> torch.Tensor:
  """
  Quantize a 2D float32 tensor to Q2_K format.
  
  Args:
    tensor: Input tensor of shape (M, N) where N must be a multiple of 256
  
  Returns:
    Quantized tensor of type uint8 and shape (M, sizeof(block_q2_K) * N/256) containing the block_q2_K data
  """ 
  return quantizer_cpp.quantize_q2_k(tensor) 