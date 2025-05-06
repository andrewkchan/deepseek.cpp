#include "model.h"

#include <assert.h>
#include <cfloat>
#include <math.h>

#include "quant.h"
#include "profile.h"

#if DEBUG_MODEL
#include <fstream>
#include "fmt/format.h"
static std::map<std::string, DebugTensor> _debug_map;
std::map<std::string, DebugTensor>& debug_map_cpu() {
  return _debug_map;
}
template <typename T>
static std::vector<T> copy_debug_tensor(T* x, size_t size) {
  std::vector<T> out(size);
  for (size_t i = 0; i < size; i++) {
    out[i] = x[i];
  }
  return out;
}
template <typename T>
static void save_debug_tensor(const std::string& name, T* x, size_t size) {
  _debug_map[name] = DebugTensor(copy_debug_tensor<T>(x, size));
}
static void dump_debug_map(const std::string& filename) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    fprintf(stderr, "Failed to open %s for writing\n", filename.c_str());
    return;
  }

  // Write Python imports
  out << "import torch\n\n";
  out << "debug_tensors = {\n";

  // Iterate through debug map and write each tensor
  bool first = true;
  for (const auto& pair : _debug_map) {
    if (!first) {
      out << ",\n";
    }
    first = false;

    const std::string& name = pair.first;
    const DebugTensor& tensor = pair.second;

    out << "    '" << name << "': torch.tensor([";

    // Write tensor values
    bool first_val = true;
    assert(tensor.data_type == DebugTensor::DataType::F32);
    for (const auto& val : tensor.data_f32) {
      if (!first_val) {
        out << ", ";
      }
      first_val = false;
      
      // Use scientific notation with high precision
      out << std::scientific << std::setprecision(8) << val;
    }
    
    out << "])";
  }
  
  out << "\n}\n";
  out.close();
}
#endif

static void matmul(
  float* xout, float* x, float* w, int n, int d, 
  const int* block_size, float* scale,
  void* unused_aqb
) {
  // W (d,n) @ x (n,) -> xout (d,)
  (void)unused_aqb;
  static float one = 1.0f;
  int dummy_block_size[2] = {d, n};
  if (scale == nullptr) {
    scale = &one;
    block_size = dummy_block_size;
  }
  int scale_num_cols = (n + block_size[1] - 1) / block_size[1];
  for (int scale_i = 0; scale_i < cdiv(d, block_size[0]); scale_i++) {
    int ii;
#pragma omp parallel for private(ii)
    for (ii = 0; ii < block_size[0]; ii++) {
      int i = scale_i * block_size[0] + ii;
      if (i >= d) {
        continue;
      }
      float val = 0.0f;
      for (int scale_j = 0; scale_j < cdiv(n, block_size[1]); scale_j++) {
        float scale_val = scale[scale_i * scale_num_cols + scale_j];
        for (int jj = 0; jj < block_size[1]; jj++) {
          int j = scale_j * block_size[1] + jj;
          if (j >= n) {
            break;
          }
          val += (w[i * n + j] * x[j]) * scale_val;
        }
      }
      xout[i] = val;
    }
  }
}

// matmul supporting float16 weights via the F16C extension, which allows
// conversion into float32 values before calculations.
static void matmul(
  float* xout, float* x, f16_t* w, int n, int d, 
  const int* block_size, float* scale,
  void* unused_aqb
) {
  (void)unused_aqb;
#if defined(__AVX2__) && defined(__F16C__)
  // W (d,n) @ x (n,) -> xout (d,)
  assert(n % 16 == 0);
  assert(scale == nullptr || block_size[1] % 16 == 0);
  static float one = 1.0f;
  int dummy_block_size[2] = {d, n};
  if (scale == nullptr) {
    scale = &one;
    block_size = dummy_block_size;
  }
  int scale_num_cols = (n + block_size[1] - 1) / block_size[1];
  for (int scale_i = 0; scale_i < cdiv(d, block_size[0]); scale_i++) {
    int ii;
#pragma omp parallel for private(ii)
    for (ii = 0; ii < block_size[0]; ii++) {
      int i = scale_i * block_size[0] + ii;
      if (i >= d) {
        continue;
      }
      // Vectorized dot product of w[i][:] and x[:] where w is a packed float16 array.
      __m256 sumlo = _mm256_setzero_ps();
      __m256 sumhi = _mm256_setzero_ps();
      for (int scale_j = 0; scale_j < cdiv(n, block_size[1]); scale_j++) {
        // Broadcast scale_val to all elements of a vector
        float scale_val = scale[scale_i * scale_num_cols + scale_j];
        __m256 scale_vec = _mm256_set1_ps(scale_val);
        for (int jj = 0; jj < block_size[1]; jj+=16) {
          int j = scale_j * block_size[1] + jj;
          if (j >= n) {
            break;
          }
          
          // Extract the next set of 16 float16 weights from `w` and store them
          // to two separate float32 vectors of width 8 (`wveclo_ps`, `wvechi_ps`)
          __m256i wvec = _mm256_loadu_si256((__m256i*)&w[i * n + j]);
          __m128i wveclo = _mm256_extractf128_si256(wvec, 0);
          __m128i wvechi = _mm256_extractf128_si256(wvec, 1);
          __m256 wveclo_ps = _mm256_cvtph_ps(wveclo);
          __m256 wvechi_ps = _mm256_cvtph_ps(wvechi);
          
          // Scale the weight vectors
          wveclo_ps = _mm256_mul_ps(wveclo_ps, scale_vec);
          wvechi_ps = _mm256_mul_ps(wvechi_ps, scale_vec);
          
          // Extract the next two float32 vectors of width 8 `xveclo`, `xvechi` from `x`
          __m256 xveclo = _mm256_loadu_ps(&x[j]);
          __m256 xvechi = _mm256_loadu_ps(&x[j + 8]);
          
          // Compute vectorized FMAs: sumlo += wveclo * xveclo, sumhi += wvechi * xvechi
          sumlo = _mm256_fmadd_ps(wveclo_ps, xveclo, sumlo);
          sumhi = _mm256_fmadd_ps(wvechi_ps, xvechi, sumhi);
        }
      }
      // Horizontally reduce width-8 float32 vectors sumlo, sumhi to a scalar.
      __m256 sum8 = _mm256_add_ps(sumlo, sumhi);              // sum8[0:8] = sumlo[0:8] + sumhi[0:8]
      __m128 sum4 = _mm_add_ps(                               // sum4[0:4] = sum8[0:4] + sum8[4:8]
        _mm256_extractf128_ps(sum8, 0), 
        _mm256_extractf128_ps(sum8, 1)
      );
      __m128 sum1 = _mm_dp_ps(sum4, _mm_set1_ps(1.0f), 0xf1); // sum1[0] = dot(sum4, [1,1,1,1])
      xout[i] = _mm_cvtss_f32(sum1);
    }
  }
#else
  assert(false && "float16 not supported on this platform");
#endif
}

// matmul supporting float8e5m2 weights via AVX2 and F16C extensions, which (1) 
// allows vectorized conversion from f8e5m2 to float16 and (2) conversion from 
// float16 to float32 values before calculations.
static void matmul(
  float* xout, float* x, f8e5m2_t* w, int n, int d, 
  const int* block_size, float* scale,
  void* unused_aqb
) {
  (void)unused_aqb;
#if defined(__AVX2__) && defined(__F16C__)
  // W (d,n) @ x (n,) -> xout (d,)
  assert(n % 16 == 0);
  assert(scale == nullptr || block_size[1] % 16 == 0);
  static float one = 1.0f;
  int dummy_block_size[2] = {d, n};
  if (scale == nullptr) {
    scale = &one;
    block_size = dummy_block_size;
  }
  int scale_num_cols = (n + block_size[1] - 1) / block_size[1];
  for (int scale_i = 0; scale_i < cdiv(d, block_size[0]); scale_i++) {
    int ii;
#pragma omp parallel for private(ii)
    for (ii = 0; ii < block_size[0]; ii++) {
      int i = scale_i * block_size[0] + ii;
      if (i >= d) {
        continue;
      }
      // Vectorized dot product of w[i][:] and x[:] where w is a packed float8e5m2 array.
      __m256 sumlo = _mm256_setzero_ps();
      __m256 sumhi = _mm256_setzero_ps();
      for (int scale_j = 0; scale_j < cdiv(n, block_size[1]); scale_j++) {
        // Broadcast scale_val to all elements of a vector
        float scale_val = scale[scale_i * scale_num_cols + scale_j];
        __m256 scale_vec = _mm256_set1_ps(scale_val);
        for (int jj = 0; jj < block_size[1]; jj+=16) {
          int j = scale_j * block_size[1] + jj;
          if (j >= n) {
            break;
          }

          // Extract the next set of 16 float8e5m2 weights from `w` and store them
          // to two separate float32 vectors of width 8 (`wveclo_ps`, `wvechi_ps`)
          __m128i wvec = _mm_loadu_si128((__m128i*)&w[i * n + j]);
          // Take each half of `wvec` which consists of 8 float8e5m2 weights and
          // pad each 8-bit float8e5m2 value with 8 zeros in the mantissa (least significant bits),
          // converting to 8 float16 values.
          __m128i wveclo = _mm_unpacklo_epi8(_mm_setzero_si128(), wvec);
          __m128i wvechi = _mm_unpackhi_epi8(_mm_setzero_si128(), wvec);
          // Widen each 8xf16 vector to 8xf32.
          __m256 wveclo_ps = _mm256_cvtph_ps(wveclo);
          __m256 wvechi_ps = _mm256_cvtph_ps(wvechi);
          
          // Scale the weight vectors
          wveclo_ps = _mm256_mul_ps(wveclo_ps, scale_vec);
          wvechi_ps = _mm256_mul_ps(wvechi_ps, scale_vec);
          
          // Extract the next two float32 vectors of width 8 `xveclo`, `xvechi` from `x`
          __m256 xveclo = _mm256_loadu_ps(&x[j]);
          __m256 xvechi = _mm256_loadu_ps(&x[j + 8]);
          // Compute vectorized FMAs: sumlo += wveclo * xveclo, sumhi += wvechi * xvechi
          sumlo = _mm256_fmadd_ps(wveclo_ps, xveclo, sumlo);
          sumhi = _mm256_fmadd_ps(wvechi_ps, xvechi, sumhi);
        }
      }
      // Horizontally reduce width-8 float32 vectors sumlo, sumhi to a scalar.
      __m256 sum8 = _mm256_add_ps(sumlo, sumhi);              // sum8[0:8] = sumlo[0:8] + sumhi[0:8]
      __m128 sum4 = _mm_add_ps(                               // sum4[0:4] = sum8[0:4] + sum8[4:8]
        _mm256_extractf128_ps(sum8, 0), 
        _mm256_extractf128_ps(sum8, 1)
      );
      __m128 sum1 = _mm_dp_ps(sum4, _mm_set1_ps(1.0f), 0xf1); // sum1[0] = dot(sum4, [1,1,1,1])
      xout[i] = _mm_cvtss_f32(sum1);
    }
  }
#else
  assert(false && "float8e5m2 not supported on this platform");
#endif
}

static void matmul(
  float* xout, float* x, block_q2_K* w, int n, int d, 
  const int* unused_block_size, float* unused_scale,
  void* aqb
) {
  // W (d,n) @ x (n,) -> xout (d,)
  (void)unused_block_size;
  (void)unused_scale;
  size_t blocks_per_row = n / QK_K;
  block_q8_K* aqb_q8 = (block_q8_K*)aqb;
  int chunk_size = QK_K * 2;
  int num_chunks = cdiv(n, chunk_size);
  {
    PROFILE_BLOCK("quantize_acts");
    #pragma omp parallel for
      for (int i = 0; i < num_chunks; i++) {
        int start = i * chunk_size;
        int k = (i == num_chunks - 1) ? (n - start) : chunk_size;
        if (k > 0) {
          quantize_row_q8_K_ref(x + start, aqb_q8 + (start/QK_K), k);
        }
      }
  }
  {
    PROFILE_BLOCK("matmul_w2a8");
    int i;
  #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
      ggml_vec_dot_q2_K_q8_K(n, xout + i, w + i * blocks_per_row, aqb_q8);
    }
  }
}

static void matmul(
  float* xout, float* x, block_q3_K* w, int n, int d, 
  const int* unused_block_size, float* unused_scale,
  void* aqb
) {
  // W (d,n) @ x (n,) -> xout (d,)
  (void)unused_block_size;
  (void)unused_scale;
  size_t blocks_per_row = n / QK_K;
  block_q8_K* aqb_q8 = (block_q8_K*)aqb;
  int chunk_size = QK_K * 2;
  int num_chunks = cdiv(n, chunk_size);
  {
    PROFILE_BLOCK("quantize_acts");
    #pragma omp parallel for
      for (int i = 0; i < num_chunks; i++) {
        int start = i * chunk_size;
        int k = (i == num_chunks - 1) ? (n - start) : chunk_size;
        if (k > 0) {
          quantize_row_q8_K_ref(x + start, aqb_q8 + (start/QK_K), k);
        }
      }
  }
  {
    PROFILE_BLOCK("matmul_w3a8");
    int i;
  #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
      ggml_vec_dot_q3_K_q8_K(n, xout + i, w + i * blocks_per_row, aqb_q8);
    }
  }
}


// Compute the softmax of an input vector `x` of length `size` and store it in `o`.
static void softmax(float* o, float* x, int size) {
  float score_max = -FLT_MAX;
  for (int i = 0; i < size; ++i) {
    if (x[i] > score_max) {
      score_max = x[i];
    }
  }
  float score_sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    o[i] = expf(x[i] - score_max);
    score_sum += o[i];
  }
  for (int i = 0; i < size; ++i) {
    o[i] /= score_sum;
  }
}

inline float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

static void moe_gate(
  float* moe_weights,
  float* moegate_bias,
  int* active_experts,
  float* x,
  int n_routed_experts,
  int n_active_routed,
  bool norm_topk_prob,
  float routed_scaling_factor,
  ScoringFunc scoring_func,
  TopKMethod topk_method,
  int n_group,
  int topk_group
) {
  // Set moe_weights[:n_active_routed] to the weights of the top K experts.
  // Set active_experts[:n_active_routed] to the indices of the top K experts.
  if (scoring_func == ScoringFunc::SOFTMAX) {
    softmax(x, x, n_routed_experts);
  } else if (scoring_func == ScoringFunc::SIGMOID) {
    for (int i = 0; i < n_routed_experts; i++) {
      x[i] = sigmoid(x[i]);
    }
  }

  if (moegate_bias) {
    for (int i = 0; i < n_routed_experts; ++i) {
      x[i] += moegate_bias[i];
    }
  }

  // top k
  float wsum = 0.0f;
  if (topk_method == TopKMethod::GREEDY) {
    assert(n_routed_experts <= 256);
    std::array<uint8_t, 32> mask{};
    for (int k = 0; k < n_active_routed; ++k) {
      int best = -1;
      for (int j = 0; j < n_routed_experts; ++j) {
        int mask_i = j / 8;
        int mask_r = j % 8;
        if ((mask[mask_i] & (1ull << mask_r)) == 0 && (best == -1 || x[j] > x[best])) {
          best = j;
        }
      }

      active_experts[k] = best;
      wsum += x[active_experts[k]];
      int best_mask_i = best / 8;
      int best_mask_r = best % 8;
      mask[best_mask_i] |= 1ull << best_mask_r;
    }
  } else if (topk_method == TopKMethod::GROUP_LIMITED_GREEDY) {
    int group_size = n_routed_experts / n_group;
    
    // First pass: select topk_group within each group
    std::array<uint8_t, 32> mask{};
    
    for (int g = 0; g < n_group; g++) {
      // Select topk_group items from this group
      for (int k = 0; k < topk_group; k++) {
        int best = -1;
        for (int j = g*group_size; j < (g+1)*group_size; j++) {
          int mask_i = j / 8;
          int mask_r = j % 8;
          if ((mask[mask_i] & (1u << mask_r)) == 0 && x[j] > x[best]) {
            best = j;
          }
        }
        int best_mask_i = best / 8;
        int best_mask_r = best % 8;
        mask[best_mask_i] |= 1u << best_mask_r;
      }
    }
    // Flip mask so that now we only look at the topk_group items in each group
    for (int i = 0; i < 32; i++) {
      mask[i] = ~mask[i];
    }
    
    // Second pass: select top n_active_routed overall
    for (int k = 0; k < n_active_routed; ++k) {
      int best = -1;
      for (int j = 0; j < n_routed_experts; ++j) {
        int mask_i = j / 8;
        int mask_r = j % 8;
        if ((mask[mask_i] & (1ull << mask_r)) == 0 && (best == -1 || x[j] > x[best])) {
          best = j;
        }
      }

      active_experts[k] = best;
      wsum += x[active_experts[k]];
      int best_mask_i = best / 8;
      int best_mask_r = best % 8;
      mask[best_mask_i] |= 1ull << best_mask_r;
    }
  } else if (topk_method == TopKMethod::NOAUX_TC) {
    assert(false && "TODO: implement noaux_tc");
  }

  if (!norm_topk_prob) {
    wsum = 1.0;
  }
  for (int k = 0; k < n_active_routed; ++k) {
    moe_weights[k] = x[active_experts[k]] / wsum * routed_scaling_factor;
  }
}

static void rmsnorm(float* o, float* x, float* weight, int size, float eps) {
  float rms = 0.0f;
  for (int i = 0; i < size; ++i) {
    rms += x[i] * x[i];
  }
  rms = sqrtf(rms / size + eps);
  float scale = 1.0f / rms;
  for (int i = 0; i < size; ++i) {
    o[i] = x[i] * scale * weight[i];
  }
}

[[maybe_unused]] static void layernorm(float* o, float* x, float* weight, float* bias, int size, float eps) {
  float mean = 0.0f;
  for (int i = 0; i < size; ++i) {
    mean += x[i];
  }
  mean /= size;
  float var = 0.0f;
  for (int i = 0; i < size; ++i) {
    var += (x[i] - mean) * (x[i] - mean);
  }
  var /= size;
  float scale = 1.0f / sqrtf(var + eps);
  if (bias) {
    for (int i = 0; i < size; ++i) {
      o[i] = (x[i] - mean) * scale * weight[i] + bias[i];
    }
  } else {
    for (int i = 0; i < size; ++i) {
      o[i] = (x[i] - mean) * scale * weight[i];
    }
  }
}

inline float gelu(float x) {
  return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

inline float silu(float x) {
  return x / (1.0f + expf(-x));
}

inline float clip(float x, float v) {
  return x < -v ? -v : (x > v ? v : x);
}

static void rope(float* buf, float* vec, int d, int head_dim, int pos, float theta) {
  // For some reason, DeepSeek-V2 was trained using rope output 
  // layout transposed compared to the input. This means we need a buffer
  // to hold intermediate results.
  assert(d % 2 == 0);
  for (int i = 0; i < d; i += 2) {
    int j_head = i % head_dim;
    float freq = 1.0f / powf(theta, (float)j_head / (float)head_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    float v0 = vec[i];
    float v1 = vec[i + 1];
    buf[i/2] = v0 * fcr - v1 * fci;
    buf[i/2 + d/2] = v0 * fci + v1 * fcr;
  }
  for (int i = 0; i < d; i++) {
    vec[i] = buf[i];
  }
}

static void rope_v3(float* vec, int d, int head_dim, int pos, float theta) {
  int rotary_dim = head_dim;

  for (int i = 0; i < d; i += 2) {
    int j_head = i % head_dim;
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    float v0 = vec[i];
    float v1 = vec[i + 1];
    vec[i] = v0 * fcr - v1 * fci;
    vec[i + 1] = v0 * fci + v1 * fcr;
  }
}

static void rope(float* buf, f16_t* vec, int d, int head_dim, int pos, float theta) {
  // For some reason, DeepSeek-V2 was trained using rope output
  // layout transposed compared to the input. This means we need a buffer
  // to hold intermediate results.
  assert(d % 2 == 0);
  for (int i = 0; i < d; i += 2) {
    int j_head = i % head_dim;
    float freq = 1.0f / powf(theta, (float)j_head / (float)head_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    float v0 = half_to_float(vec[i]);
    float v1 = half_to_float(vec[i + 1]);
    buf[i/2] = v0 * fcr - v1 * fci;
    buf[i/2 + d/2] = v0 * fci + v1 * fcr;
  }
  for (int i = 0; i < d; i++) {
    vec[i] = float_to_half(buf[i]);
  }
}

static void rope_v3(f16_t* vec, int d, int head_dim, int pos, float theta) {
  int rotary_dim = head_dim;

  for (int i = 0; i < d; i += 2) {
    int j_head = i % head_dim;
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    float v0 = half_to_float(vec[i]);
    float v1 = half_to_float(vec[i + 1]);
    vec[i] = float_to_half(v0 * fcr - v1 * fci);
    vec[i + 1] = float_to_half(v0 * fci + v1 * fcr);
  }
}


// Compute next value in a sequence for a single causal self-attention head.
void attn(
  float* xout,    // (n_heads * v_head_dim,) - output vector
  float* atth,    // (kv_len,) - scratch space to hold attention scores of the sequence
  const float* qh,      // (head_dim,) - query vector for this head
  const f16_t* kh,      // (kv_len, n_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
  const f16_t* vh,      // (kv_len, n_heads, v_head_dim) - buffer containing value vectors of the sequence for all KV heads
  int head_dim,   // size of the "key-space"
  int v_head_dim, // size of the "value-space"
  int n_heads, // number of attention heads
  int kv_len      // number of tokens of the sequence we will attend over
) {
  int k_stride = n_heads * head_dim; // stride per token in this k head
  // calculate attention scores as dot products of q and k
  for (int t = 0; t < kv_len; ++t) {
    float score = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      score += qh[i] * half_to_float(kh[t * k_stride + i]);
    }
    score /= sqrtf(head_dim);
    atth[t] = score;
  }

  // softmax the scores to get attention weights over [0..kv_len)
  softmax(atth, atth, kv_len);

  int v_stride = n_heads * v_head_dim; // stride per token in this v head
  // mix values with attention weights
  for (int i = 0; i < v_head_dim; ++i) {
    float vi = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
      vi += atth[t] * half_to_float(vh[t * v_stride + i]);
    }
    xout[i] = vi;
  }
}

// Compute next value in a sequence for a single causal self-attention head.
// MLA variant: uses combined latent-KV cache and PE-KV cache.
void attn_mla(
  float* xout,    // (n_heads * kv_lora_rank,) - output vector
  float* atth,    // (kv_len,) - scratch space to hold attention scores of the sequence
  const float* qh_c,    // (kv_lora_rank,) - transformed latent query vector for this head
  const float* qh_rope, // (qk_rope_head_dim,) - PE-query vector for this head
  const f16_t* compressed_kv,      // (kv_len, kv_lora_rank) - buffer containing latent vectors of the sequence
  const f16_t* k_rope,  // (kv_len, qk_rope_head_dim) - buffer containing PE key-vectors of the sequence
  int head_dim, // used for softmax scale factor
  int kv_lora_rank, // size of the "latent-space"
  int qk_rope_head_dim, // size of the "PE-space"
  int kv_len   // number of tokens of the sequence we will attend over
) {
  int kv_stride = kv_lora_rank; // stride per token in the latent buffer
  int k_rope_stride = qk_rope_head_dim; // stride per token in the PE buffer
  // calculate attention scores as dot products of q and k
  for (int t = 0; t < kv_len; ++t) {
    float score = 0.0f;
    for (int i = 0; i < kv_lora_rank; ++i) {
      score += qh_c[i] * half_to_float(compressed_kv[t * kv_stride + i]);
    }
    for (int i = 0; i < qk_rope_head_dim; ++i) {
      score += qh_rope[i] * half_to_float(k_rope[t * k_rope_stride + i]);
    }
    score /= sqrtf(head_dim);
    atth[t] = score;
  }

  // softmax the scores to get attention weights over [0..kv_len)
  softmax(atth, atth, kv_len);

  // mix latents with attention weights
  for (int i = 0; i < kv_lora_rank; ++i) {
    float vi = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
      vi += atth[t] * half_to_float(compressed_kv[t * kv_stride + i]);
    }
    xout[i] = vi;
  }
}

// Compute forward pass for a single block and update the inference state accordingly.
// PRECONDITIONS: 
// - `s.x()` contains the input to the block. Output will also go here.
// - Block KV cache is hydrated.
template <typename T>
void Block::_block_cpu(
  InferenceState& s,  // inference state
  int pos,            // index of the current token in the sequence
  int kv_sink,        // number of sink tokens currently in the KV cache
  int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
  int kv_len          // number of tokens in the kv cache that we will attend over
) const {
  const Config& c = *_config;

  // Attention pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.xb(), s.x(), rms_att_weight(), c.dim, c.norm_eps);
      break;
    }
  }

  // Attention output into `hb`
  attention_impl(s, pos, kv_sink, kv_pos, kv_len);

  // Residual back into `x`
  for (int i = 0; i < c.dim; ++i) {
    s.x()[i] += s.hb()[i];
  }

  // FFN pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.xb(), s.x(), rms_ffn_weight(), c.dim, c.norm_eps);
      break;
    }
  }

  if (c.n_routed_experts > 0 && moegate() != nullptr) {
    PROFILE_BLOCK(ffn_moe);
    // Block is a sparse MoE FFN layer
    PROFILE(matmul_unscaled(s.moe_weights(), s.xb(), moegate(), c.dim, c.n_routed_experts));
    moe_gate(
      s.active_experts_weights(), _moegate_bias, s.active_experts(), s.moe_weights(),
      c.n_routed_experts, c.n_active_routed, c.norm_topk_prob, c.routed_scaling_factor,
      c.scoring_func, c.topk_method, c.n_group, c.topk_group
    );
    for (int k = 0; k < c.n_active_routed; ++k) {
      int expert_index = s.active_experts()[k];
      size_t expert_size = c.dim * c.moe_intermediate_size;
      int expert_scale13_size = _s1 ? cdiv(c.moe_intermediate_size, c.block_size[0]) * cdiv(c.dim, c.block_size[1]) : 0;
      int expert_scale2_size = _s2 ? cdiv(c.dim, c.block_size[0]) * cdiv(c.moe_intermediate_size, c.block_size[1]) : 0;
      size_t weight_offset = expert_index * expert_size;
      if (is_k_quant(c.weight_quant)) {
        // In K-quants, each element of the weight tensor is a block of QK_K elements
        weight_offset = weight_offset / QK_K;
      }
      size_t scale13_offset = expert_index * expert_scale13_size;
      size_t scale2_offset = expert_index * expert_scale2_size;
      // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
      // Note this is a feedforward with a GLU, not a simple MLP.
      PROFILE(matmul(s.hb(), s.xb(), w1<T>() + weight_offset, c.dim, c.moe_intermediate_size, c.block_size.data(), _s1 + scale13_offset, s.aqb()));
      PROFILE(matmul(s.hb2(), s.xb(), w3<T>() + weight_offset, c.dim, c.moe_intermediate_size, c.block_size.data(), _s3 + scale13_offset, s.aqb()));
      switch (c.act) {
        case ActivationType::GELU: {
          for (int i = 0; i < c.moe_intermediate_size; ++i) {
            s.hb()[i] = gelu(s.hb()[i]) * s.hb2()[i];
          }
          break;
        }
        case ActivationType::SILU: {
          for (int i = 0; i < c.moe_intermediate_size; ++i) {
            s.hb()[i] = silu(s.hb()[i]) * s.hb2()[i];
          }
          break;
        }
      }
      PROFILE(matmul(s.xb2(), s.hb(), w2<T>() + weight_offset, c.moe_intermediate_size, c.dim, c.block_size.data(), _s2 + scale2_offset, s.aqb()));
      float expert_weight = s.active_experts_weights()[k];
      for (int i = 0; i < c.dim; ++i) {
        s.x()[i] += s.xb2()[i] * expert_weight;
      }
    }
    if (c.n_shared_experts > 0) {
      // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
      // Note this is a feedforward with a GLU, not a simple MLP.
      PROFILE(matmul(s.hb(), s.xb(), shared_w1<T>(), c.dim, c.n_shared_experts * c.moe_intermediate_size, c.block_size.data(), _shared_s1, s.aqb()));
      PROFILE(matmul(s.hb2(), s.xb(), shared_w3<T>(), c.dim, c.n_shared_experts * c.moe_intermediate_size, c.block_size.data(), _shared_s3, s.aqb()));
      switch (c.act) {
        case ActivationType::GELU: {
          for (int i = 0; i < c.n_shared_experts * c.moe_intermediate_size; ++i) {
            s.hb()[i] = gelu(s.hb()[i]) * s.hb2()[i];
          }
          break;
        }
        case ActivationType::SILU: {
          for (int i = 0; i < c.n_shared_experts * c.moe_intermediate_size; ++i) {
            s.hb()[i] = silu(s.hb()[i]) * s.hb2()[i];
          }
          break;
        }
      }

      PROFILE(matmul(s.xb2(), s.hb(), shared_w2<T>(), c.n_shared_experts * c.moe_intermediate_size, c.dim, c.block_size.data(), _shared_s2, s.aqb()));
      // residual connection back into x
      for (int i = 0; i < c.dim; ++i) {
        s.x()[i] += s.xb2()[i];
      }
    }
  } else {
    PROFILE_BLOCK(ffn_dense);
    // Block is a dense FFN layer
    // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
    // Note this is a feedforward with a GLU, not a simple MLP.
    PROFILE(matmul(s.hb(), s.xb(), w1<T>(), c.dim, c.hidden_dim, c.block_size.data(), _s1, s.aqb()));
    PROFILE(matmul(s.hb2(), s.xb(), w3<T>(), c.dim, c.hidden_dim, c.block_size.data(), _s3, s.aqb()));
    switch (c.act) {
      case ActivationType::GELU: {
        for (int i = 0; i < c.hidden_dim; ++i) {
          s.hb()[i] = gelu(s.hb()[i]) * s.hb2()[i];
        }
        break;
      }
      case ActivationType::SILU: {
        for (int i = 0; i < c.hidden_dim; ++i) {
          s.hb()[i] = silu(s.hb()[i]) * s.hb2()[i];
        }
        break;
      }
    }
    PROFILE(matmul(s.xb2(), s.hb(), w2<T>(), c.hidden_dim, c.dim, c.block_size.data(), _s2, s.aqb()));
    // residual connection back into x
    for (int i = 0; i < c.dim; ++i) {
      s.x()[i] += s.xb2()[i];
    }
  }
}

template<typename T>
void BlockMHA::_attention_impl(
  InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len
) const {
  PROFILE_BLOCK(attn_mha);
  const Config& c = *_config;
  int q_dim = c.n_heads * c.head_dim;

  // qkv matmuls for this position
  if (c.q_lora_rank > 0) {
    PROFILE(matmul(s.q_a(), s.xb(), this->wq_a<T>(), c.dim, c.q_lora_rank, c.block_size.data(), _sq_a, s.aqb()));
    switch (c.norm_type) {
      case LayerNormType::RMSNorm: {
        rmsnorm(s.q_a(), s.q_a(), this->rms_q_a_weight(), c.q_lora_rank, c.norm_eps);
        break;
      }
    }
    PROFILE(matmul(s.q(), s.q_a(), this->wq_b<T>(), c.q_lora_rank, q_dim, c.block_size.data(), _sq_b, s.aqb()));
  } else {
    PROFILE(matmul(s.q(), s.xb(), this->wq<T>(), c.dim, q_dim, c.block_size.data(), _sq, s.aqb()));
  }
  PROFILE(matmul(s.kv_a(), s.xb(), this->wkv_a<T>(), c.dim, c.kv_lora_rank + c.qk_rope_head_dim, c.block_size.data(), _skv_a, s.aqb()));

  // Apply RoPE positional encoding
  int q_pe_offset = c.head_dim - c.qk_rope_head_dim;
  bool is_v3 = c.has_moegate_bias;
  for (int h = 0; h < c.n_heads; h++) {
    if (is_v3) {
      rope_v3(s.q(h) + q_pe_offset, c.qk_rope_head_dim, c.qk_rope_head_dim, pos, c.rope_theta);
    } else {
      rope(s.ropebuf(), s.q(h) + q_pe_offset, c.qk_rope_head_dim, c.qk_rope_head_dim, pos, c.rope_theta);
    }
  }
  int kv_pe_offset = c.kv_lora_rank;
  float* k_rope = s.kv_a() + kv_pe_offset;
  if (is_v3) {
    rope_v3(k_rope, c.qk_rope_head_dim, c.qk_rope_head_dim, pos, c.rope_theta);
  } else {
    rope(s.ropebuf(), k_rope, c.qk_rope_head_dim, c.qk_rope_head_dim, pos, c.rope_theta);
  }
  // rms norm to non-pe chunk of kv_a
  rmsnorm(s.kv_a(), s.kv_a(), this->rms_kv_a_weight(), c.kv_lora_rank, c.norm_eps);
  // un-compress the latent kv via multiplication with wkv_b
  int qk_nope_head_dim = c.head_dim - c.qk_rope_head_dim;
  int uncompressed_kv_dim = c.n_heads * (qk_nope_head_dim + c.v_head_dim);
  PROFILE(matmul(s.kv_b(), s.kv_a(), this->wkv_b<T>(), c.kv_lora_rank, uncompressed_kv_dim, c.block_size.data(), _skv_b, s.aqb()));
  // concatenate kv_b and k_rope in each head to build key heads
  for (int h = 0; h < c.n_heads; h++) {
    for (int i = 0; i < qk_nope_head_dim; i++) {
      s.k(h)[i] = s.kv_b(h)[i];
    }
    for (int i = 0; i < c.qk_rope_head_dim; i++) {
      s.k(h)[qk_nope_head_dim + i] = k_rope[i];
    }
  }
  // transfer value heads from kv_b
  for (int h = 0; h < c.n_heads; h++) {
    for (int i = 0; i < c.v_head_dim; i++) {
      s.v(h)[i] = s.kv_b(h)[qk_nope_head_dim + i];
    }
  }

  // update kv cache
  int key_dim = c.n_heads * c.head_dim;
  for (int i = 0; i < key_dim; ++i) {
    this->key_cache(kv_pos)[i] = float_to_half(s.k()[i]);
  }
  int value_dim = c.n_heads * c.v_head_dim;
  for (int i = 0; i < value_dim; ++i) {
    this->value_cache(kv_pos)[i] = float_to_half(s.v()[i]);
  }

  // Sink tokens remain untouched while the rest of the KV cache is incrementally 
  // replaced in ring order, but sink i must always be positioned (max_seq_len - i)
  // away from current timestep. Hence, each forward pass, rotate sink tokens 
  // forward by 1. See https://arxiv.org/abs/2309.17453 for more.
  for (int r = 0; r < kv_sink; r++) {
    f16_t* key = key_cache(r);
    // in-place update PE-chunk of each key head
    int q_pe_offset = c.head_dim - c.qk_rope_head_dim;
    for (int h = 0; h < c.n_heads; h++) {
      f16_t* kh = key + h * c.head_dim;
      if (is_v3) {
        rope_v3(kh + q_pe_offset, c.qk_rope_head_dim, c.qk_rope_head_dim, 1, c.rope_theta);
      } else {
        rope(s.ropebuf(), kh + q_pe_offset, c.qk_rope_head_dim, c.qk_rope_head_dim, 1, c.rope_theta);
      }
    }
  }

  {
    PROFILE_BLOCK(self_attn_mha_inner);
    f16_t* kb = this->key_cache();
    f16_t* vb = this->value_cache();
    int h;
  #pragma omp parallel for private(h)
    for (h = 0; h < c.n_heads; h++) {
      int k_head_offset = h * c.head_dim;
      int v_head_offset = h * c.v_head_dim;
      f16_t* kh = kb + k_head_offset; // Use pointer arithmetic for base address
      f16_t* vh = vb + v_head_offset; // Use pointer arithmetic for base address
      attn(
        s.xb2(h, c.v_head_dim), // Output per Q head
        s.att(h),              // Attention scores per Q head
        s.q(h),                // Query vector for this head
        kh,                    // Pointer to start of relevant K cache base
        vh,                    // Pointer to start of relevant V cache base
        c.head_dim,            // Dimension of K space
        c.v_head_dim,          // Dimension of V space
        c.n_heads,          // Total number of KV heads (passed to inner attn func for stride calculation)
        kv_len                 // Sequence length to attend over
      );
    }
  }

  // final matmul to get output of the attention, place result in s.hb() for residual connection
  PROFILE(matmul(s.hb(), s.xb2(), this->wo<T>(), c.n_heads * c.v_head_dim, c.dim, c.block_size.data(), _so, s.aqb()));
}

template<typename T>
void BlockMLA::_attention_impl(
  InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len
) const {
  PROFILE_BLOCK(attn_mla);
  const Config& c = *_config;
  assert(c.q_lora_rank > 0); // MLA requires q_lora_rank > 0

  // qkv down projections
  PROFILE(matmul(s.q_a(), s.xb(), this->wq_a<T>(), c.dim, c.q_lora_rank, c.block_size.data(), _sq_a, s.aqb()));
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.q_a(), s.q_a(), this->rms_q_a_weight(), c.q_lora_rank, c.norm_eps);
      break;
    }
  }
  PROFILE(matmul(s.kv_a(), s.xb(), this->wkv_a<T>(), c.dim, c.kv_lora_rank + c.qk_rope_head_dim, c.block_size.data(), _skv_a, s.aqb()));
  // query transformations
  PROFILE(matmul(s.q_rope(), s.q_a(), this->wq_rope_b<T>(), c.q_lora_rank, c.n_heads * c.qk_rope_head_dim, c.block_size.data(), _sq_rope_b, s.aqb()));
  PROFILE(matmul(s.q_c(), s.q_a(), this->wc<T>(), c.q_lora_rank, c.n_heads * c.kv_lora_rank, c.block_size.data(), _sc, s.aqb()));

  // Apply RoPE positional encoding
  bool is_v3 = c.has_moegate_bias;
  for (int h = 0; h < c.n_heads; h++) {
    if (is_v3) {
      rope_v3(s.q_rope(h), c.qk_rope_head_dim, c.qk_rope_head_dim, pos, c.rope_theta);
    } else {
      rope(s.ropebuf(), s.q_rope(h), c.qk_rope_head_dim, c.qk_rope_head_dim, pos, c.rope_theta);
    }
  }
  int kv_pe_offset = c.kv_lora_rank;
  float* k_rope = s.kv_a() + kv_pe_offset;
  if (is_v3) {
    rope_v3(k_rope, c.qk_rope_head_dim, c.qk_rope_head_dim, pos, c.rope_theta);
  } else {
    rope(s.ropebuf(), k_rope, c.qk_rope_head_dim, c.qk_rope_head_dim, pos, c.rope_theta);
  }
  // rms norm to non-pe chunk of kv_a (compressed latent kv)
  rmsnorm(s.kv_a(), s.kv_a(), this->rms_kv_a_weight(), c.kv_lora_rank, c.norm_eps);

  // update kv cache
  for (int i = 0; i < c.kv_lora_rank; ++i) {
    this->kv_nope_cache(kv_pos)[i] = float_to_half(s.kv_a()[i]);
  }
  for (int i = 0; i < c.qk_rope_head_dim; ++i) {
    this->kv_rope_cache(kv_pos)[i] = float_to_half(k_rope[i]);
  }

  // Sink tokens remain untouched while the rest of the KV cache is incrementally 
  // replaced in ring order, but sink i must always be positioned (max_seq_len - i)
  // away from current timestep. Hence, each forward pass, rotate sink tokens 
  // forward by 1. See https://arxiv.org/abs/2309.17453 for more.
  for (int r = 0; r < kv_sink; r++) {
    f16_t* kv = this->kv_rope_cache(r);
    if (is_v3) {
      rope_v3(kv, c.qk_rope_head_dim, c.qk_rope_head_dim, 1, c.rope_theta);
    } else {
      rope(s.ropebuf(), kv, c.qk_rope_head_dim, c.qk_rope_head_dim, 1, c.rope_theta);
    }
  }

  {
    PROFILE_BLOCK(self_attn_mla_inner);
    int h;
  #pragma omp parallel for private(h)
    for (h = 0; h < c.n_heads; h++) {
      attn_mla(
        s.xb2(h, c.kv_lora_rank), // Output is per-head latent vector
        s.att(h),
        s.q_c(h),
        s.q_rope(h),
        this->kv_nope_cache(),
        this->kv_rope_cache(),
        c.head_dim,
        c.kv_lora_rank,
        c.qk_rope_head_dim,
        kv_len
      );
    }
  }

  // Uncompress latent kvs output by each attention head, storing result in `kv_b`.
  // We reuse kv_b buffer here for the uncompressed value outputs.
  for (int h = 0; h < c.n_heads; h++) {
    float* v_b_head = s.kv_b() + h * c.v_head_dim;
    size_t v_b_head_size = c.v_head_dim * c.kv_lora_rank;
    size_t v_b_head_offset = v_b_head_size * h;
    if (is_k_quant(c.weight_quant)) {
      // In k-quants, each element of the weight tensor is a block of QK_K elements
      v_b_head_offset = v_b_head_offset / QK_K;
    }
    T* wv_b_head = wv_b<T>() + v_b_head_offset;
    PROFILE(matmul(v_b_head, s.xb2(h, c.kv_lora_rank), wv_b_head, c.kv_lora_rank, c.v_head_dim, c.block_size.data(), _sv_b, s.aqb()));
  }

  // final matmul to get output of the attention, place result in s.hb() for residual connection
  PROFILE(matmul(s.hb(), s.kv_b(), this->wo<T>(), c.n_heads * c.v_head_dim, c.dim, c.block_size.data(), _so, s.aqb()));
}

void mha_cpu(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  f16_t* kb,    // (max_seq_len, n_heads, head_dim)
  f16_t* vb,    // (max_seq_len, n_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int v_head_dim, int kv_len, int max_seq_len, int n_heads
) {
  // Multihead attention. Iterate over all heads.
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < n_heads; h++) {
    int k_head_offset = h * head_dim;
    int v_head_offset = h * v_head_dim;
    f16_t* kh = kb + k_head_offset;
    f16_t* vh = vb + v_head_offset;
    attn(
      xout + head_dim * h, att + max_seq_len * h, q + head_dim * h, 
      kh, vh, head_dim, v_head_dim, n_heads, kv_len
    );
  }
}

void matmul_unscaled(float* xout, float* x, float* w, int n, int d) {
  matmul(xout, x, w, n, d, nullptr, nullptr, nullptr);
}
void matmul_unscaled(float* xout, float* x, f16_t* w, int n, int d) {
  matmul(xout, x, w, n, d, nullptr, nullptr, nullptr);
}
void matmul_unscaled(float* xout, float* x, f8e5m2_t* w, int n, int d) {
  matmul(xout, x, w, n, d, nullptr, nullptr, nullptr);
}

void ffn_cpu(
  float* xout, float* x, 
  float* w1, float* w2, float* w3, 
  int hidden_dim, int dim,
  ActivationType act
) {
  float* hb = new float[hidden_dim];
  float* hb2 = new float[hidden_dim];
  // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
  // Note this is a feedforward with a GLU, not a simple MLP.
  matmul_unscaled(hb, x, w1, dim, hidden_dim);
  matmul_unscaled(hb2, x, w3, dim, hidden_dim);
  switch (act) {
    case ActivationType::GELU: {
      for (int i = 0; i < hidden_dim; ++i) {
        hb[i] = gelu(hb[i]) * hb2[i];
      }
      break;
    }
    case ActivationType::SILU: {
      for (int i = 0; i < hidden_dim; ++i) {
        hb[i] = silu(hb[i]) * hb2[i];
      }
      break;
    }
  }

  matmul_unscaled(xout, hb, w2, hidden_dim, dim);
  
  delete[] hb;
  delete[] hb2;
}

template void Block::_block_cpu<float>(InferenceState&, int, int, int, int) const;
template void Block::_block_cpu<f16_t>(InferenceState&, int, int, int, int) const;
template void Block::_block_cpu<f8e5m2_t>(InferenceState&, int, int, int, int) const;
template void Block::_block_cpu<block_q2_K>(InferenceState&, int, int, int, int) const;
template void Block::_block_cpu<block_q3_K>(InferenceState&, int, int, int, int) const;

template void BlockMHA::_attention_impl<float>(InferenceState&, int, int, int, int) const;
template void BlockMHA::_attention_impl<f16_t>(InferenceState&, int, int, int, int) const;
template void BlockMHA::_attention_impl<f8e5m2_t>(InferenceState&, int, int, int, int) const;
template void BlockMHA::_attention_impl<block_q2_K>(InferenceState&, int, int, int, int) const;
template void BlockMHA::_attention_impl<block_q3_K>(InferenceState&, int, int, int, int) const;

template void BlockMLA::_attention_impl<float>(InferenceState&, int, int, int, int) const;
template void BlockMLA::_attention_impl<f16_t>(InferenceState&, int, int, int, int) const;
template void BlockMLA::_attention_impl<f8e5m2_t>(InferenceState&, int, int, int, int) const;
template void BlockMLA::_attention_impl<block_q2_K>(InferenceState&, int, int, int, int) const;
template void BlockMLA::_attention_impl<block_q3_K>(InferenceState&, int, int, int, int) const;

void Model::_copy_embedding(InferenceState& s, int token) {
  const Config& c = *config;
  switch (c.weight_quant) {
    case Quant::F32: {
      float* emb = static_cast<float*>(token_embedding_table);
      for (int i = 0; i < c.dim; ++i) {
        s.x()[i] = emb[token * c.dim + i];
      }
      break;
    }
    case Quant::F16: {
      f16_t* emb = static_cast<f16_t*>(token_embedding_table);
      for (int i = 0; i < c.dim; i+=1) {
        s.x()[i] = half_to_float(emb[token * c.dim + i]);
      }
      break;
    }
    case Quant::F8E5M2: {
      f8e5m2_t* emb = static_cast<f8e5m2_t*>(token_embedding_table);
      int* block_size = config->block_size.data();
      int scale_num_cols = (c.dim + block_size[1] - 1) / block_size[1];
      for (int i = 0; i < c.dim; i+=1) {
        int scale_i = token / block_size[0];
        int scale_j = i / block_size[1];
        float scale = token_embedding_scale[scale_i * scale_num_cols + scale_j];
        s.x()[i] = float8e5m2_to_float(emb[token * c.dim + i]) * scale;
      }
      break;
    }
    case Quant::Q2_K: {
      block_q2_K* emb = static_cast<block_q2_K*>(token_embedding_table);
      int blocks_per_row = c.dim / QK_K;
      dequantize_row_q2_K(emb + token * blocks_per_row, s.x(), c.dim);
      break;
    }
    case Quant::Q3_K: {
      block_q3_K* emb = static_cast<block_q3_K*>(token_embedding_table);
      int blocks_per_row = c.dim / QK_K;
      dequantize_row_q3_K(emb + token * blocks_per_row, s.x(), c.dim);
      break;
    }
    default: {
      assert(false && "unsupported weight quantization");
    }
  }
}

void Model::_forward_cpu(InferenceState& s, int token, int pos, InferenceMode mode) {
  const Config& c = *config;

  // copy the token embedding into `x`
  PROFILE(_copy_embedding(s, token));

  // When decoding past the context length, keep the first few tokens in the KV cache
  // untouched as "attention sinks" while replacing the rest in ring order.
  // See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
  int kv_sink = pos >= c.max_seq_len ? KV_SINKS : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (c.max_seq_len - kv_sink);
  int kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;

  // forward all layers in order
  for (auto b : blocks) {
    b->block(s, pos, kv_sink, kv_pos, kv_len);
  }

  if (mode == InferenceMode::HYDRATE_KV_CACHE) {
    // only hydrate the KV cache and don't compute output logits
    return;
  }

  // final layer norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.x(), s.x(), rms_final_weight, c.dim, c.norm_eps);
      break;
    }
  }

  // classifier into logits
  {
    PROFILE_BLOCK(lm_head);
    switch (c.weight_quant) {
      case Quant::F32: {
        matmul_unscaled(s.logits(), s.x(), static_cast<float*>(wcls), c.dim, c.vocab_size);
        break;
      }
      case Quant::F16: {
        matmul_unscaled(s.logits(), s.x(), static_cast<f16_t*>(wcls), c.dim, c.vocab_size);
        break;
      }
      case Quant::F8E5M2: {
        matmul(s.logits(), s.x(), static_cast<f8e5m2_t*>(wcls), c.dim, c.vocab_size, c.block_size.data(), scls, s.aqb());
        break;
      }
      case Quant::Q2_K: {
        matmul(s.logits(), s.x(), static_cast<block_q2_K*>(wcls), c.dim, c.vocab_size, c.block_size.data(), scls, s.aqb());
        break;
      }
      case Quant::Q3_K: {
        matmul(s.logits(), s.x(), static_cast<block_q3_K*>(wcls), c.dim, c.vocab_size, c.block_size.data(), scls, s.aqb());
        break;
      }
      default: {
        assert(false && "unsupported weight quantization");
      }
    }
  }
}