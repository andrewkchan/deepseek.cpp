#pragma once

#include "codec.h"
#include "cuda_runtime_api.h"

#include <memory>
#include <vector>
#include <map>

#define DEBUG_MODEL 0

constexpr int KV_SINKS = 2;

enum class ActivationType {
  GELU,
  SILU,
};

enum class LayerNormType {
  RMSNorm,
};

enum class Device {
  CPU,
  CUDA,
};

enum class InferenceMode {
  HYDRATE_KV_CACHE, // only hydrate the KV cache and don't compute output logits
  OUTPUT_LOGITS // set InferenceState logits to logits for the next token
};

extern "C" void* upload_cuda(void* host, size_t size);
extern "C" void* download_cuda(void* device, size_t size, std::string debug);
extern "C" void register_cuda_host(void* host, size_t size);
extern "C" void free_cuda(void* device);
extern "C" void unregister_cuda_host(void* host);
extern "C" void set_cuda_device(int device);
extern "C" void init_cuda_stream(cudaStream_t* stream);

struct Config {
  int dim;                  // transformer input & output dimension
  int hidden_dim;           // dimension of hidden layer in feedforward network (dense blocks only)
  int n_layers;             // number of layers
  int n_heads;              // number of attention query heads
  int n_kv_heads;           // number of key and value heads; can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
  int vocab_size;           // vocabulary size
  int max_seq_len;          // max sequence length
  float rope_theta;         // RoPE theta
  float norm_eps;           // epsilon for layer normalization
  ActivationType act;       // activation function
  LayerNormType norm_type;  // norm type
  int first_k_dense_replace; // how many blocks to keep the dense FFN (when sparse MoE is default)
  // mixture of experts
  int n_shared_experts;
  int n_routed_experts;
  int n_active_routed;
  int moe_intermediate_size;
  float routed_scaling_factor;
  bool norm_topk_prob;
  int topk_group;
  // multi-latent attention
  int kv_lora_rank;
  int q_lora_rank;
  int qk_nope_head_dim;
  int qk_rope_head_dim;
  int v_head_dim;
  int head_dim;             // dimension of each attention head, equal to qk_nope_head_dim + qk_rope_head_dim
  // Data type of the weights according to config, used
  // to safety check tensor dtype at initialization time.
  DType weight_dtype;

  // If nonzero `context` is supplied, max sequence length is limited to `context`.
  void from_yalm(YALMData& yalm, int context = 0);
  size_t active_bytes(size_t pos) const;
};

struct CudaGraph {
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  bool is_created = false;
  std::unordered_map<std::string, cudaGraphNode_t> nodes;

  void wrap(std::function<void()> func, cudaStream_t s);
  void launch(cudaStream_t s);
  void add_or_update_kernel_node(std::string key, cudaKernelNodeParams params, cudaStream_t stream);
};

// Buffer for all state used during a forward pass.
// Members are reused across subsequent blocks and passes.
// This lets us avoid allocations during inference.
struct InferenceState {
  InferenceState(const std::shared_ptr<Config> config);
  ~InferenceState();

  // current activations
  float* x() const { return _x; }
  float* xb() const { return _xb; }
  float* xb(int head) const { return _xb + _config->head_dim * head; }
  // TODO: do we need xb2?
  float* xb2() const { return _xb2; }
  float* xb2(int head, int head_size) const { return _xb2 + head_size * head; }
  float* hb() const { return _hb; }
  float* hb2() const { return _hb2; }
  float* q_a() const { return _q_a; }
  float* q() const { return _q; }
  float* q(int head) const { return _q + _config->head_dim * head; }
  float* kv_a() const { return _kv_a; }
  float* kv_b() const { return _kv_b; }
  float* kv_b(int head) const { return _kv_b + (_config->head_dim - _config->qk_rope_head_dim + _config->v_head_dim) * head; }
  float* ropebuf() const { return _ropebuf; }
  float* k() const { return _k; }
  float* k(int head) const { return _k + _config->head_dim * head; }
  float* v() const { return _v; }
  float* v(int head) const { return _v + _config->v_head_dim * head; }
  float* att() const { return _att; }
  float* att(int head) const { return _att + _config->max_seq_len * head; }
  // mixture of experts
  float* moe_weights() const { return _moe_weights; }
  float* active_experts_weights() const { return _active_experts_weights; }
  int* active_experts() const { return _active_experts; }
  // LM head
  float* logits() const { return _logits; }

  void cuda();
  Device device() const { return _device; }
  cudaStream_t stream() const { return _stream; }
  InferenceMode mode() const { return _mode; }
  void set_mode(InferenceMode mode) { _mode = mode; }
  CudaGraph& graph() {
    return _mode == InferenceMode::HYDRATE_KV_CACHE ? _hydrate_graph : _output_graph;
  }

private:
  std::shared_ptr<Config> _config;
  Device _device = Device::CPU;
  cudaStream_t _stream;
  InferenceMode _mode = InferenceMode::OUTPUT_LOGITS;
  CudaGraph _hydrate_graph;
  CudaGraph _output_graph;

  // current activations
  float* _x = nullptr;         // (dim,) - latest activation
  float* _xb = nullptr;        // (dim,) - activation inside a residual branch
  float* _xb2 = nullptr;       // (max{dim, n_kv_heads * v_head_dim},) - activation inside a residual branch (second slot)
  float* _hb = nullptr;        // (hidden_dim,) - buffer for hidden dimension in feedforward network
  float* _hb2 = nullptr;       // (hidden_dim,) - buffer for hidden dimension in feedforward network (second slot)
  float* _q_a = nullptr;       // (q_lora_rank,) - compressed (latent) query vector for latest timestamp
  float* _q = nullptr;         // (n_heads * head_dim,) - query vectors for latest timestamp
  float* _kv_a = nullptr;      // (kv_lora_rank + qk_rope_head_dim,) - compressed (latent) key-value vector for latest timestamp
  float* _kv_b = nullptr;      // (n_kv_heads * (head_dim-qk_rope_head_dim+v_head_dim),) - uncompressed key-value vector for latest timestamp
  float* _ropebuf = nullptr;   // (n_kv_heads * qk_rope_head_dim,) - buffer for rope
  float* _k = nullptr;         // (n_kv_heads * head_dim,) - key vectors for latest timestamp
  float* _v = nullptr;         // (n_kv_heads * v_head_dim,) - value vectors for latest timestamp
  float* _att = nullptr;       // (n_heads, seq_len) - buffer for attention scores
  // mixture of experts
  float* _moe_weights = nullptr; // (n_routed_experts,) - buffer for expert weights, decided by router
  float* _active_experts_weights = nullptr; // (n_active_experts,) - buffer for weights of top K experts (active experts)
  int* _active_experts = nullptr; // (n_active_experts,) - buffer for indices of top K experts (active experts)
  
  // LM head
  // NOTE: this always lives on the host (CPU), but must be registered 
  // with CUDA to be used on the device.
  float* _logits = nullptr;    // (vocab_size,) - final output logits
};

/* Transformer Block */
struct Block {
  Block(
    int layer_i,
    const std::shared_ptr<Config> config,
    const Tensor* rms_att_weight,
    const Tensor* rms_q_a_weight,
    const Tensor* rms_kv_a_weight,
    const Tensor* rms_ffn_weight,
    const Tensor* wq,
    const Tensor* wq_a,
    const Tensor* wq_b,
    const Tensor* wkv_a,
    const Tensor* wkv_b,
    const Tensor* wo,
    const Tensor* w1,
    const Tensor* w2,
    const Tensor* w3,
    const Tensor* shared_w1,
    const Tensor* shared_w2,
    const Tensor* shared_w3,
    const Tensor* moegate
  );
  ~Block();

  float* rms_att_weight() const { return _rms_att_weight; }
  float* rms_ffn_weight() const { return _rms_ffn_weight; }
  float* rms_q_a_weight() const { return _rms_q_a_weight; }
  float* rms_kv_a_weight() const { return _rms_kv_a_weight; }
  template <typename T>
  T* wq() const { return static_cast<T*>(_wq); }
  template <typename T>
  T* wq_a() const { return static_cast<T*>(_wq_a); }
  template <typename T>
  T* wq_b() const { return static_cast<T*>(_wq_b); }
  template <typename T>
  T* wkv_a() const { return static_cast<T*>(_wkv_a); }
  template <typename T>
  T* wkv_b() const { return static_cast<T*>(_wkv_b); }
  template <typename T>
  T* wo() const { return static_cast<T*>(_wo); }
  template <typename T>
  T* w1() const { return static_cast<T*>(_w1); }
  template <typename T>
  T* w2() const { return static_cast<T*>(_w2); }
  template <typename T>
  T* w3() const { return static_cast<T*>(_w3); }
  template <typename T>
  T* moegate() const { return static_cast<T*>(_moegate); }
  template <typename T>
  T* shared_w1() const { return static_cast<T*>(_shared_w1); }
  template <typename T>
  T* shared_w2() const { return static_cast<T*>(_shared_w2); }
  template <typename T>
  T* shared_w3() const { return static_cast<T*>(_shared_w3); }
  f16_t* key_cache() const { return _key_cache; }
  f16_t* key_cache(int pos) const { return _key_cache + pos * _config->head_dim * _config->n_kv_heads; }
  f16_t* value_cache() const { return _value_cache; }
  f16_t* value_cache(int pos) const { return _value_cache + pos * _config->v_head_dim * _config->n_kv_heads; }

  // Compute forward pass for this block and update the inference state accordingly.
  // PRECONDITIONS: 
  // - `s.x()` contains the input to the block. Output will also go here.
  // - Block KV cache is hydrated.
  void block(
    InferenceState& s,  // inference state
    int pos,            // index of the current token in the sequence
    int kv_sink,        // number of sink tokens currently in the KV cache
    int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
    int kv_len          // number of tokens in the kv cache that we will attend over
  ) const;

  void cuda();

private:
  template <typename T>
  void _block_cpu(
    InferenceState& s,  // inference state
    int pos,            // index of the current token in the sequence
    int kv_sink,        // number of sink tokens currently in the KV cache
    int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
    int kv_len          // number of tokens in the kv cache that we will attend over
  ) const;
  template <typename T>
  void _block_cuda(
    InferenceState& s,  // inference state
    int pos,            // index of the current token in the sequence
    int kv_sink,        // number of sink tokens currently in the KV cache
    int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
    int kv_len          // number of tokens in the kv cache that we will attend over
  ) const;

  int _layer_i = 0;

  std::shared_ptr<Config> _config;
  Device _device = Device::CPU;

  // weights for norms
  float* _rms_att_weight = nullptr; // (dim) rmsnorm weights
  float* _rms_q_a_weight = nullptr; // (q_lora_rank) rmsnorm weights
  float* _rms_kv_a_weight = nullptr; // (kv_lora_rank + qk_rope_head_dim)
  float* _rms_ffn_weight = nullptr; // (dim)

  // weights for self-attention matmuls
  void* _wq = nullptr; // (n_heads * head_dim, dim)
  void* _wq_a = nullptr; // (q_lora_rank, dim)
  void* _wq_b = nullptr; // (n_heads * head_dim, q_lora_rank)
  void* _wkv_a = nullptr; // (kv_lora_rank + qk_rope_head_dim, dim)
  void* _wkv_b = nullptr; // (n_kv_heads * (head_dim-qk_rope_head_dim+v_head_dim), kv_lora_rank)
  void* _wo = nullptr; // (dim, n_heads * v_head_dim)
  
  // weights for ffn
  void* _w1 = nullptr; // (n_routed_experts?, moe_intermediate_size, dim) or (hidden_dim, dim)
  void* _w2 = nullptr; // (n_routed_experts?, dim, moe_intermediate_size) or (dim, hidden_dim)
  void* _w3 = nullptr; // (n_routed_experts?, moe_intermediate_size, dim) or (hidden_dim, dim)
  void* _shared_w1 = nullptr; // (n_shared_experts?, moe_intermediate_size, dim)
  void* _shared_w2 = nullptr; // (n_shared_experts?, dim, moe_intermediate_size)
  void* _shared_w3 = nullptr; // (n_shared_experts?, moe_intermediate_size, dim)
  // weights for mixture of experts router if present
  void* _moegate = nullptr; // (n_routed_experts?, dim)

  // kv cache
  f16_t* _key_cache = nullptr;   // (seq_len, n_kv_heads * head_dim)
  f16_t* _value_cache = nullptr; // (seq_len, n_kv_heads * v_head_dim)
};

struct Model {
  std::shared_ptr<Config> config;

  std::vector<std::shared_ptr<Block>> blocks;
  
  // token embedding table
  void* token_embedding_table = nullptr; // (vocab_size, dim)
  // final norm
  float* rms_final_weight = nullptr; // (dim,)
  // classifier weights for the logits, on the last layer
  void* wcls = nullptr; // (vocab_size, dim)

  Model(YALMData& yalm, int context = 0);
  
  void forward(InferenceState& s, int token, int pos, InferenceMode mode = InferenceMode::OUTPUT_LOGITS);
  void cuda();

private:
  void _forward_cpu(InferenceState& s, int token, int pos, InferenceMode mode);
  void _forward_cuda(InferenceState& s, int token, int pos, InferenceMode mode);
  void _forward_cuda_build_graph(InferenceState& s, int token, int pos, InferenceMode mode);
  void _copy_embedding(InferenceState& s, int token);

  Device _device = Device::CPU;
};

#if DEBUG_MODEL
struct DebugTensor {
  enum struct DataType {
    F32,
    F16,
  };

  DebugTensor() = default;
  DebugTensor(const std::vector<float>& data);
  DebugTensor(const std::vector<f16_t>& data);
  DebugTensor& operator=(const DebugTensor& other) = default;
  float max_err(const DebugTensor& other) const;

  std::vector<float> data_f32;
  std::vector<f16_t> data_f16;
  DataType data_type;
};
std::map<std::string, DebugTensor>& debug_map_cpu();
std::map<std::string, DebugTensor>& debug_map_cuda();
#endif

////////////////////////////////////////
// Exposed for tests
////////////////////////////////////////
void attn(
  float* xout,    // (dim,) - output vector
  float* atth,    // (kv_len,) - scratch space to hold attention scores of the sequence
  float* qh,      // (head_dim,) - query vector for this head
  f16_t* kh,      // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
  f16_t* vh,      // (kv_len, n_kv_heads, head_dim) - buffer containing value vectors of the sequence for all KV heads
  int head_dim,   // size of the "key-space"
  int v_head_dim, // size of the "value-space"
  int n_kv_heads, // number of kv heads, can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
  int kv_len      // number of tokens of the sequence we will attend over
);

void mha_cpu(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  f16_t* kb,    // (max_seq_len, n_kv_heads, head_dim)
  f16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int v_head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
);
// TODO update me for MLA
void mha_cuda(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  f16_t* kb,    // (max_seq_len, n_kv_heads, head_dim)
  f16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
);

void matmul_cpu(float* xout, float* x, float* w, int n, int d);
void matmul_cpu(float* xout, float* x, f16_t* w, int n, int d);
template <typename T>
void matmul_cuda(float* xout, float* x, T* w, int n, int d);

void ffn_cpu(
  float* xout, float* x, 
  float* w1, float* w2, float* w3, 
  int hidden_dim, int dim,
  ActivationType act
);
template <typename T>
void ffn_cuda(
  float* xout, float* x, 
  T* w1, T* w2, T* w3, 
  int hidden_dim, int dim,
  ActivationType act
);
////////////////////////////////////////