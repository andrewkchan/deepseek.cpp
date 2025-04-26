#include <torch/extension.h>
#include <vector>
#include "quant.h"

torch::Tensor quantize_q2_k(torch::Tensor& input) {
  // Row-major quantization (equivalent to block size [1, 256]) 
  // of input tensor using Q2_K scheme.
  TORCH_CHECK(input.ndimension() == 2, "input must be 2D");
  TORCH_CHECK(input.size(1) % QK_K == 0, "ncols must be divisible by QK_K");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }
  const int64_t nrows = input.size(0);
  const int64_t ncols = input.size(1);
  const int64_t blocks_per_row = ncols / QK_K;
  const int64_t block_size = sizeof(block_q2_K);
  
  auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto output = torch::empty({nrows, blocks_per_row * block_size}, options);
  
  const float* input_ptr = input.data_ptr<float>();
  uint8_t* output_ptr = output.data_ptr<uint8_t>();

  // Parallelize over rows
  #pragma omp parallel for
  for (int64_t row = 0; row < nrows; row++) {
    const float* row_input = input_ptr + row * ncols;
    block_q2_K* row_output = reinterpret_cast<block_q2_K*>(output_ptr + row * blocks_per_row * block_size);

    quantize_row_q2_K_ref(row_input, row_output, ncols);
  }
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_q2_k", &quantize_q2_k, "Quantize a tensor to Q2_K format");
} 