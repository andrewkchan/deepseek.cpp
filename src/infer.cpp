#include "model.h"

#include <assert.h>
#include <cfloat>
#include <math.h>

#include "immintrin.h"
#include "f16cintrin.h"

#if defined(__AVX2__) && defined(__F16C__)
inline float half_to_float(f16_t x) {
  return _cvtsh_ss(x);
}
inline f16_t float_to_half(float x) {
  return _cvtss_sh(x, 0);
}
#else
inline float half_to_float(f16_t x) {
  assert(false && "float16 not supported on this platform");
  return 0.0f;
}
inline f16_t float_to_half(float x) {
  assert(false && "float16 not supported on this platform");
  return 0;
}
#endif

#if DEBUG_MODEL
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
#endif

static void matmul(float* xout, float* x, float* w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}

// matmul supporting float16 weights via the F16C extension, which allows
// conversion into float32 values before calculations.
static void matmul(float* xout, float* x, f16_t* w, int n, int d) {
#if defined(__AVX2__) && defined(__F16C__)
  // W (d,n) @ x (n,) -> xout (d,)
  assert(n % 16 == 0);
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    // Vectorized dot product of w[i][:] and x[:] where w is a packed float16 array.
    __m256 sumlo = _mm256_setzero_ps();
    __m256 sumhi = _mm256_setzero_ps();
    for (int j = 0; j < n; j+=16) {
      // Extract the next set of 16 float16 weights from `w` and store them
      // to two separate float32 vectors of width 8 (`wveclo_ps`, `wvechi_ps`)
      __m256i wvec = _mm256_loadu_si256((__m256i*)&w[i * n + j]);
      __m128i wveclo = _mm256_extractf128_si256(wvec, 0);
      __m128i wvechi = _mm256_extractf128_si256(wvec, 1);
      __m256 wveclo_ps = _mm256_cvtph_ps(wveclo);
      __m256 wvechi_ps = _mm256_cvtph_ps(wvechi);
      // Extract the next two float32 vectors of width 8 `xveclo`, `xvechi` from `x`
      __m256 xveclo = _mm256_loadu_ps(&x[j]);
      __m256 xvechi = _mm256_loadu_ps(&x[j + 8]);
      // Compute vectorized FMAs: sumlo += wveclo * xveclo, sumhi += wvechi * xvechi
      sumlo = _mm256_fmadd_ps(wveclo_ps, xveclo, sumlo);
      sumhi = _mm256_fmadd_ps(wvechi_ps, xvechi, sumhi);
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
#else
  assert(false && "float16 not supported on this platform");
#endif
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

static void moe_gate(
  float* moe_weights, 
  int* active_experts, 
  float* x, 
  int n_routed_experts, 
  int n_active_routed, 
  bool norm_topk_prob,
  float routed_scaling_factor
) {
  // Set moe_weights[:n_active_routed] to the weights of the top K experts.
  // Set active_experts[:n_active_routed] to the indices of the top K experts.
  softmax(x, x, n_routed_experts);
  
  // top k
  assert(n_routed_experts <= 256);
  std::array<uint8_t, 32> mask{};
  float wsum = 0.0f;
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

// Compute next value in a sequence for a single causal self-attention head.
void attn(
  float* xout,    // (n_kv_heads * v_head_dim,) - output vector
  float* atth,    // (kv_len,) - scratch space to hold attention scores of the sequence
  float* qh,      // (head_dim,) - query vector for this head
  f16_t* kh,      // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
  f16_t* vh,      // (kv_len, n_kv_heads, v_head_dim) - buffer containing value vectors of the sequence for all KV heads
  int head_dim,   // size of the "key-space"
  int v_head_dim, // size of the "value-space"
  int n_kv_heads, // number of kv heads, can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
  int kv_len      // number of tokens of the sequence we will attend over
) {
  int k_stride = n_kv_heads * head_dim; // stride per token in this k head
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

  int v_stride = n_kv_heads * v_head_dim; // stride per token in this v head
  // mix values with attention weights
  for (int i = 0; i < v_head_dim; ++i) {
    float vi = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
      vi += atth[t] * half_to_float(vh[t * v_stride + i]);
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
  assert(_config);
  const Config& c = *_config;

  // attention pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.xb(), s.x(), rms_att_weight(), c.dim, c.norm_eps);
      break;
    }
  }

  int q_dim = c.n_heads * c.head_dim;

  // qkv matmuls for this position
  if (c.q_lora_rank > 0) {
    matmul(s.q_a(), s.xb(), wq_a<T>(), c.dim, c.q_lora_rank);
    switch (c.norm_type) {
      case LayerNormType::RMSNorm: {
        rmsnorm(s.q_a(), s.q_a(), rms_q_a_weight(), c.q_lora_rank, c.norm_eps);
        break;
      }
    }
    matmul(s.q(), s.q_a(), wq_b<T>(), c.q_lora_rank, q_dim);
  } else {
    matmul(s.q(), s.xb(), wq<T>(), c.dim, q_dim);
  }
  matmul(s.kv_a(), s.xb(), wkv_a<T>(), c.dim, c.kv_lora_rank + c.qk_rope_head_dim);

  // Apply RoPE positional encoding to the PE chunks of q and kv_a
  int q_pe_offset = c.head_dim - c.qk_rope_head_dim;
  for (int h = 0; h < c.n_heads; h++) {
    rope(s.ropebuf(), s.q(h) + q_pe_offset, c.qk_rope_head_dim, c.qk_rope_head_dim, pos, c.rope_theta);
  }
  int kv_pe_offset = c.kv_lora_rank;
  float* k_rope = s.kv_a() + kv_pe_offset;
  rope(s.ropebuf(), k_rope, c.qk_rope_head_dim, c.qk_rope_head_dim, pos, c.rope_theta);
  // rms norm to non-pe chunk of kv_a (compressed latent kv)
  rmsnorm(s.kv_a(), s.kv_a(), rms_kv_a_weight(), c.kv_lora_rank, c.norm_eps);
  // un-compress the latent kv via multiplication with wkv_b
  int qk_nope_head_dim = c.head_dim - c.qk_rope_head_dim;
  int uncompressed_kv_dim = c.n_kv_heads * (qk_nope_head_dim + c.v_head_dim);
  matmul(s.kv_b(), s.kv_a(), wkv_b<T>(), c.kv_lora_rank, uncompressed_kv_dim);
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
  int key_dim = c.n_kv_heads * c.head_dim;
  for (int i = 0; i < key_dim; ++i) {
    key_cache(kv_pos)[i] = float_to_half(s.k()[i]);
  }
  int value_dim = c.n_kv_heads * c.v_head_dim;
  for (int i = 0; i < value_dim; ++i) {
    value_cache(kv_pos)[i] = float_to_half(s.v()[i]);
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
      rope(s.ropebuf(), kh + q_pe_offset, c.qk_rope_head_dim, c.qk_rope_head_dim, 1, c.rope_theta);
    }
  }

  // Multihead attention. Iterate over all heads.
  f16_t* kb = key_cache();
  f16_t* vb = value_cache();
  int q_per_kv_head = c.n_heads / c.n_kv_heads; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < c.n_heads; h++) {
    int k_head_offset = (h / q_per_kv_head) * c.head_dim;
    int v_head_offset = (h / q_per_kv_head) * c.v_head_dim;
    f16_t* kh = kb + k_head_offset;
    f16_t* vh = vb + v_head_offset;
    attn(s.xb2(h, c.v_head_dim), s.att(h), s.q(h), kh, vh, c.head_dim, c.v_head_dim, c.n_kv_heads, kv_len);
  }

  // final matmul to get output of the attention, using `hb` as temp storage
  matmul(s.hb(), s.xb2(), wo<T>(), c.n_kv_heads * c.v_head_dim, c.dim);

  // residual connection back into x
  for (int i = 0; i < c.dim; ++i) {
    s.x()[i] += s.hb()[i];
  }
  
  // ffn pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.xb(), s.x(), rms_ffn_weight(), c.dim, c.norm_eps);
      break;
    }
  }

  if (c.n_routed_experts > 0 && moegate<T>() != nullptr) {
    // Block is a sparse MoE FFN layer
    matmul(s.moe_weights(), s.xb(), moegate<T>(), c.dim, c.n_routed_experts);
    moe_gate(
      s.active_experts_weights(), s.active_experts(), s.moe_weights(), 
      c.n_routed_experts, c.n_active_routed, c.norm_topk_prob, c.routed_scaling_factor
    );
    for (int k = 0; k < c.n_active_routed; ++k) {
      int expert_index = s.active_experts()[k];
      int expert_size = c.dim * c.moe_intermediate_size;
      // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
      // Note this is a feedforward with a GLU, not a simple MLP.
      matmul(s.hb(), s.xb(), w1<T>() + expert_index * expert_size, c.dim, c.moe_intermediate_size);
      matmul(s.hb2(), s.xb(), w3<T>() + expert_index * expert_size, c.dim, c.moe_intermediate_size);
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
      matmul(s.xb2(), s.hb(), w2<T>() + expert_index * expert_size, c.moe_intermediate_size, c.dim);
      float expert_weight = s.active_experts_weights()[k];
      // residual connection back into x
      for (int i = 0; i < c.dim; ++i) {
        s.x()[i] += s.xb2()[i] * expert_weight;
      }
    }
    if (c.n_shared_experts > 0) {
      // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
      // Note this is a feedforward with a GLU, not a simple MLP.
      matmul(s.hb(), s.xb(), shared_w1<T>(), c.dim, c.n_shared_experts * c.moe_intermediate_size);
      matmul(s.hb2(), s.xb(), shared_w3<T>(), c.dim, c.n_shared_experts * c.moe_intermediate_size);
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

      matmul(s.xb2(), s.hb(), shared_w2<T>(), c.n_shared_experts * c.moe_intermediate_size, c.dim);
      // residual connection back into x
      for (int i = 0; i < c.dim; ++i) {
        s.x()[i] += s.xb2()[i];
      }
    }
  } else {
    // Block is a dense FFN layer
    // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
    // Note this is a feedforward with a GLU, not a simple MLP.
    matmul(s.hb(), s.xb(), w1<T>(), c.dim, c.hidden_dim);
    matmul(s.hb2(), s.xb(), w3<T>(), c.dim, c.hidden_dim);
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
    matmul(s.xb2(), s.hb(), w2<T>(), c.hidden_dim, c.dim);
    // residual connection back into x
    for (int i = 0; i < c.dim; ++i) {
      s.x()[i] += s.xb2()[i];
    }
  }
}

void mha_cpu(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  f16_t* kb,    // (max_seq_len, n_kv_heads, head_dim)
  f16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int v_head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
) {
  // Multihead attention. Iterate over all heads.
  int q_per_kv_head = n_heads / n_kv_heads; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < n_heads; h++) {
    int k_head_offset = (h / q_per_kv_head) * head_dim;
    int v_head_offset = (h / q_per_kv_head) * v_head_dim;
    f16_t* kh = kb + k_head_offset;
    f16_t* vh = vb + v_head_offset;
    attn(
      xout + head_dim * h, att + max_seq_len * h, q + head_dim * h, 
      kh, vh, head_dim, v_head_dim, n_kv_heads, kv_len
    );
  }
}

void matmul_cpu(float* xout, float* x, float* w, int n, int d) {
  matmul(xout, x, w, n, d);
}
void matmul_cpu(float* xout, float* x, f16_t* w, int n, int d) {
  matmul(xout, x, w, n, d);
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
  matmul(hb, x, w1, dim, hidden_dim);
  matmul(hb2, x, w3, dim, hidden_dim);
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

  matmul(xout, hb, w2, hidden_dim, dim);
  
  delete[] hb;
  delete[] hb2;
}

template void Block::_block_cpu<float>(InferenceState&, int, int, int, int) const;
template void Block::_block_cpu<f16_t>(InferenceState&, int, int, int, int) const;

void Model::_copy_embedding(InferenceState& s, int token) {
  const Config& c = *config;
  switch (c.weight_dtype) {
    case DType::F32: {
      float* emb = static_cast<float*>(token_embedding_table);
      for (int i = 0; i < c.dim; ++i) {
        s.x()[i] = emb[token * c.dim + i];
      }
      break;
    }
    case DType::F16: {
      f16_t* emb = static_cast<f16_t*>(token_embedding_table);
      for (int i = 0; i < c.dim; i+=1) {
        s.x()[i] = half_to_float(emb[token * c.dim + i]);
      }
      break;
    }
    default: {
      assert(false && "unsupported weight dtype");
    }
  }
}

void Model::_forward_cpu(InferenceState& s, int token, int pos, InferenceMode mode) {
  const Config& c = *config;

  // copy the token embedding into `x`
  _copy_embedding(s, token);

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
  switch (c.weight_dtype) {
    case DType::F32: {
      matmul(s.logits(), s.x(), static_cast<float*>(wcls), c.dim, c.vocab_size);
      break;
    }
    case DType::F16: {
      matmul(s.logits(), s.x(), static_cast<f16_t*>(wcls), c.dim, c.vocab_size);
      break;
    }
    default: {
      assert(false && "unsupported weight dtype");
    }
  }
}