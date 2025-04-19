#include "model.h"

#include "json.hpp"
#include <algorithm>
#include <array>
#include <cfloat>
#include "fmt/format.h"
#include <iostream>
#include <limits.h>
#include <string>

#include "immintrin.h"

#include "quant.h"

using json = nlohmann::json;

int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

void Config::from_yalm(YALMData& yalm, int context) {
  dim = std::stoi(yalm.metadata.at("dim").get<std::string>());
  hidden_dim = std::stoi(yalm.metadata.at("hidden_dim").get<std::string>());
  n_layers = std::stoi(yalm.metadata.at("n_layers").get<std::string>());
  n_heads = std::stoi(yalm.metadata.at("n_heads").get<std::string>());
  n_kv_heads = std::stoi(yalm.metadata.at("n_kv_heads").get<std::string>());
  vocab_size = std::stoi(yalm.metadata.at("vocab_size").get<std::string>());
  // mixture of experts
  n_shared_experts = yalm.metadata.contains("n_shared_experts") ? std::stoi(yalm.metadata.at("n_shared_experts").get<std::string>()) : 0;
  n_routed_experts = yalm.metadata.contains("n_routed_experts") ? std::stoi(yalm.metadata.at("n_routed_experts").get<std::string>()) : 0;
  n_active_routed = yalm.metadata.contains("n_active_routed") ? std::stoi(yalm.metadata.at("n_active_routed").get<std::string>()) : 0;
  moe_intermediate_size = yalm.metadata.contains("moe_intermediate_size") ? std::stoi(yalm.metadata.at("moe_intermediate_size").get<std::string>()) : 0;
  routed_scaling_factor = yalm.metadata.contains("routed_scaling_factor") ? std::stof(yalm.metadata.at("routed_scaling_factor").get<std::string>()) : 1.0;
  n_group = yalm.metadata.contains("n_group") ? std::stoi(yalm.metadata.at("n_group").get<std::string>()) : 1;
  norm_topk_prob = yalm.metadata.contains("norm_topk_prob") ? yalm.metadata.at("norm_topk_prob").get<std::string>() == "True" : false;
  std::string scoring_func_str = yalm.metadata.value("scoring_func", "softmax");
  if (scoring_func_str == "softmax") {
    scoring_func = ScoringFunc::SOFTMAX;
  } else if (scoring_func_str == "sigmoid") {
    scoring_func = ScoringFunc::SIGMOID;
  } else {
    std::cerr << "unsupported scoring_func '" << scoring_func_str << "', defaulting to softmax" << std::endl;
    scoring_func = ScoringFunc::SOFTMAX;
  }
  topk_group = yalm.metadata.contains("topk_group") ? std::stoi(yalm.metadata.at("topk_group").get<std::string>()) : 0;
  std::string topk_method_str = yalm.metadata.value("topk_method", "");
  if (topk_method_str == "greedy") {
    topk_method = TopKMethod::GREEDY;
  } else if (topk_method_str == "group_limited_greedy") {
    topk_method = TopKMethod::GROUP_LIMITED_GREEDY;
  } else if (topk_method_str == "noaux_tc") {
    topk_method = TopKMethod::NOAUX_TC;
    assert(false && "TODO: support for Deepseek v3");
  } else {
    std::cerr << "unsupported topk_method '" << topk_method_str << "', defaulting to greedy" << std::endl;
    topk_method = TopKMethod::GREEDY;
  }
  has_moegate_bias = yalm.metadata.at("arch").get<std::string>() == "DeepseekV3ForCausalLM";
  // multi-latent attention
  kv_lora_rank = yalm.metadata.contains("kv_lora_rank") ? std::stoi(yalm.metadata.at("kv_lora_rank").get<std::string>()) : 0;
  q_lora_rank = yalm.metadata.contains("q_lora_rank") ? std::stoi(yalm.metadata.at("q_lora_rank").get<std::string>()) : 0;
  qk_nope_head_dim = yalm.metadata.contains("qk_nope_head_dim") ? std::stoi(yalm.metadata.at("qk_nope_head_dim").get<std::string>()) : 0;
  qk_rope_head_dim = yalm.metadata.contains("qk_rope_head_dim") ? std::stoi(yalm.metadata.at("qk_rope_head_dim").get<std::string>()) : 0;
  v_head_dim = yalm.metadata.contains("v_head_dim") ? std::stoi(yalm.metadata.at("v_head_dim").get<std::string>()) : 0;
  head_dim = qk_nope_head_dim + qk_rope_head_dim;

  // for now limit seq_len to 4096 to avoid KV cache OOM for models like Mistral since window size isn't correctly specified
  max_seq_len = std::min(std::stoi(yalm.metadata.at("max_seq_len").get<std::string>()), 4096);
  if (context) {
    max_seq_len = context;
  }

  rope_theta = std::stof(yalm.metadata.at("rope_theta").get<std::string>());
  norm_eps = std::stof(yalm.metadata.value("norm_eps", "1e-5"));

  std::string act_str = yalm.metadata.value("act_type", "gelu");
  if (act_str == "gelu") {
    act = ActivationType::GELU;
  } else if (act_str == "silu") {
    act = ActivationType::SILU;
  } else {
    std::cerr << "unsupported act_type, defaulting to gelu" << std::endl;
    act = ActivationType::GELU;
  }

  std::string norm_type_str = yalm.metadata.value("norm_type", "rmsnorm");
  if (norm_type_str == "rmsnorm") {
    norm_type = LayerNormType::RMSNorm;
  } else {
    std::cerr << "unsupported norm_type, defaulting to rmsnorm" << std::endl;
    norm_type = LayerNormType::RMSNorm;
  }

  first_k_dense_replace = yalm.metadata.contains("first_k_dense_replace") ? 
    std::stoi(yalm.metadata.at("first_k_dense_replace").get<std::string>()) : 0;

  std::string quant = yalm.metadata.at("quant").get<std::string>();
  if (quant == "fp32") {
    weight_quant = Quant::F32;
  } else if (quant == "fp16") {
    weight_quant = Quant::F16;
  } else if (quant == "f8e5m2") {
    weight_quant = Quant::F8E5M2;
  } else if (quant == "q2k") {
    weight_quant = Quant::Q2_K;
  } else {
    std::cerr << "FATAL: unsupported quant: " << quant << std::endl;
    assert(false);
  }

  // quantization
  if (yalm.metadata.contains("quantization_block_size_0")) {
    block_size[0] = std::stoi(yalm.metadata.at("quantization_block_size_0").get<std::string>());
    block_size[1] = std::stoi(yalm.metadata.at("quantization_block_size_1").get<std::string>());
  }
}

size_t Config::active_bytes(size_t pos) const {
  float bytes_per_weight = bits_per_weight(weight_quant, block_size[0] * block_size[1]) / 8.0;

  size_t bytes_per_block = 0;
  bytes_per_block += 2 * dim * sizeof(float); // rms_att_weight, rms_ffn_weight
  bytes_per_block += (kv_lora_rank + qk_rope_head_dim) * sizeof(float); // rms_kv_a_weight
  bytes_per_block += n_heads * head_dim * dim * bytes_per_weight; // wq
  bytes_per_block += (kv_lora_rank + qk_rope_head_dim) * dim * bytes_per_weight; // wkv_a
  bytes_per_block += n_kv_heads * (head_dim-qk_rope_head_dim+v_head_dim) * kv_lora_rank * bytes_per_weight; // wkv_b
  bytes_per_block += n_heads * v_head_dim * dim * bytes_per_weight; // wo
  if (n_routed_experts > 0) {
    bytes_per_block += n_routed_experts * dim * sizeof(float); // moegate
    bytes_per_block += n_routed_experts * sizeof(float); // moegate_bias
    bytes_per_block += n_active_routed * 3 * dim * moe_intermediate_size * bytes_per_weight; // w1, w2, w3
  } else {
    bytes_per_block += 3 * dim * hidden_dim * bytes_per_weight; // w1, w2, w3
  }
  if (n_shared_experts > 0) {
    bytes_per_block += n_shared_experts * dim * moe_intermediate_size * bytes_per_weight; // shared_w1, shared_w2, shared_w3
  }
  size_t kv_len = std::min(static_cast<size_t>(max_seq_len), pos + 1);
  size_t kv_entry_size = sizeof(f16_t);
  bytes_per_block += 2 * kv_len * n_kv_heads * head_dim * kv_entry_size; // key_cache, value_cache

  size_t bytes = 0;
  bytes += dim * bytes_per_weight; // 1 row of token_embedding_table
  bytes += n_layers * bytes_per_block; // blocks
  bytes += dim * sizeof(float); // rms_final_weight
  bytes += vocab_size * dim * sizeof(float); // wcls

  return bytes;
}

void* check_tensor(const Tensor* tensor, Quant weight_quant, std::array<int, 4> shape, const int debug_line) {
  if (tensor == nullptr) {
    std::cerr << "FATAL: missing tensor at line " << debug_line << std::endl;
    assert(false);
    return nullptr;
  }
  CodecDType expected_dtype = quant_to_codec_dtype(weight_quant);
  std::array<int, 4> expected_shape = shape;
  if (weight_quant == Quant::Q2_K) {
    size_t numel = 1;
    for (int i = 0; i < 4; i++) {
      if (shape[i] > 0) {
        numel *= shape[i];
      }
    }
    size_t total_blocks = numel / QK_K;
    size_t total_bytes = total_blocks * sizeof(block_q2_K);
    if (tensor->dtype != expected_dtype || tensor->size != total_bytes) {
      std::cerr << "FATAL: tensor mismatch for " << tensor->name << std::endl;
      std::cerr 
        << fmt::format(
          "expected: dtype={}, size={}", 
          codec_dtype_to_string(expected_dtype), 
          total_bytes
        ) 
        << std::endl;
      std::cerr 
        << fmt::format(
          "got: dtype={}, size={}", 
          codec_dtype_to_string(tensor->dtype), 
          tensor->size
        ) << std::endl;
    }
  }
  if (tensor->dtype != expected_dtype || tensor->shape != expected_shape) {
    std::cerr << "FATAL: tensor mismatch for " << tensor->name << std::endl;
    std::cerr 
      << fmt::format(
        "expected: dtype={}, shape=[{},{},{},{}]", 
        codec_dtype_to_string(expected_dtype), 
        expected_shape[0], 
        expected_shape[1], 
        expected_shape[2], 
        expected_shape[3]
      ) 
      << std::endl;
    std::cerr 
      << fmt::format(
        "got: dtype={}, shape=[{},{},{},{}]", 
        codec_dtype_to_string(tensor->dtype), 
        tensor->shape[0], tensor->shape[1], tensor->shape[2], tensor->shape[3]
      ) 
      << std::endl;
    assert(false);
  }
  return tensor->data;
};

const Tensor* get_tensor(const YALMData& yalm, const std::string& key) {
  auto it = yalm.tensors.find(key);
  if (it == yalm.tensors.end()) {
    std::cerr << "FATAL: missing tensor: " << key << std::endl;
    assert(false);
    return nullptr;
  }
  const Tensor& tensor = it->second;
  return &tensor;
};

Block::Block(
  int layer_i,
  const std::shared_ptr<Config> config,
  const Tensor* rms_att_weight,
  const Tensor* rms_q_a_weight,
  const Tensor* rms_kv_a_weight,
  const Tensor* rms_ffn_weight,
  const Tensor* wq,
  const Tensor* sq,
  const Tensor* wq_a,
  const Tensor* sq_a,
  const Tensor* wq_b,
  const Tensor* sq_b,
  const Tensor* wkv_a,
  const Tensor* skv_a,
  const Tensor* wkv_b,
  const Tensor* skv_b,
  const Tensor* wo,
  const Tensor* so,
  const Tensor* w1,
  const Tensor* s1,
  const Tensor* w2,
  const Tensor* s2,
  const Tensor* w3,
  const Tensor* s3,
  const Tensor* shared_w1,
  const Tensor* shared_s1,
  const Tensor* shared_w2,
  const Tensor* shared_s2,
  const Tensor* shared_w3,
  const Tensor* shared_s3,
  const Tensor* moegate,
  const Tensor* moegate_bias
) {
  _layer_i = layer_i;
  _config = config;
  switch (config->weight_quant) {
    case Quant::F32:
    case Quant::F16:
    case Quant::F8E5M2:
    case Quant::Q2_K: {
      break;
    }
    default: {
      std::cerr << "FATAL: unsupported weight quantization: " << quant_to_string(config->weight_quant) << std::endl;
      assert(false);
      break;
    }
  }

  _rms_att_weight = static_cast<float*>(check_tensor(
    rms_att_weight, Quant::F32, {config->dim, 0, 0, 0}, __LINE__
  ));
  if (config->q_lora_rank > 0) {
    _rms_q_a_weight = static_cast<float*>(check_tensor(
      rms_q_a_weight, Quant::F32, {config->q_lora_rank, 0, 0, 0}, __LINE__
    ));
  }
  _rms_kv_a_weight = static_cast<float*>(check_tensor(
    rms_kv_a_weight, Quant::F32, {config->kv_lora_rank, 0, 0, 0}, __LINE__
  ));
  _rms_ffn_weight = static_cast<float*>(check_tensor(
    rms_ffn_weight, Quant::F32, {config->dim, 0, 0, 0}, __LINE__
  ));

  if (config->q_lora_rank > 0) {
    _wq_a = check_tensor(
      wq_a, config->weight_quant, {config->q_lora_rank, config->dim, 0, 0}, __LINE__
    );
    _wq_b = check_tensor(
      wq_b, config->weight_quant, {config->n_heads * config->head_dim, config->q_lora_rank, 0, 0}, __LINE__
    );
  } else {
    _wq = check_tensor(
      wq, config->weight_quant, {config->n_heads * config->head_dim, config->dim, 0, 0}, __LINE__
    );
  }
  _wkv_a = check_tensor(
    wkv_a, config->weight_quant, {config->kv_lora_rank + config->qk_rope_head_dim, config->dim, 0, 0}, __LINE__
  );
  _wkv_b = check_tensor(
    wkv_b, config->weight_quant, {config->n_kv_heads * (config->head_dim-config->qk_rope_head_dim+config->v_head_dim), config->kv_lora_rank, 0, 0}, __LINE__
  );
  _wo = check_tensor(
    wo, config->weight_quant, {config->dim, config->n_heads * config->v_head_dim, 0, 0}, __LINE__
  );

  if (config->n_routed_experts > 0 && layer_i >= config->first_k_dense_replace) {
    _moegate = static_cast<float*>(check_tensor(
      moegate, Quant::F32, {config->n_routed_experts, config->dim, 0, 0}, __LINE__
    ));
    if (moegate_bias != nullptr) {
      _moegate_bias = static_cast<float*>(check_tensor(
        moegate_bias, Quant::F32, {config->n_routed_experts, 0, 0, 0}, __LINE__
      ));
    }
    _w1 = check_tensor(
      w1, config->weight_quant, {config->n_routed_experts, config->moe_intermediate_size, config->dim, 0}, __LINE__
    );
    _w2 = check_tensor(
      w2, config->weight_quant, {config->n_routed_experts, config->dim, config->moe_intermediate_size, 0}, __LINE__
    );
    _w3 = check_tensor(
      w3, config->weight_quant, {config->n_routed_experts, config->moe_intermediate_size, config->dim, 0}, __LINE__
    );
    if (config->n_shared_experts > 0) {
      _shared_w1 = check_tensor(
        shared_w1, config->weight_quant, {config->n_shared_experts * config->moe_intermediate_size, config->dim, 0}, __LINE__
      );
      _shared_w2 = check_tensor(
        shared_w2, config->weight_quant, {config->dim, config->n_shared_experts * config->moe_intermediate_size, 0}, __LINE__
      );
      _shared_w3 = check_tensor(
        shared_w3, config->weight_quant, {config->n_shared_experts * config->moe_intermediate_size, config->dim, 0}, __LINE__
      );
    }
  } else {
    _w1 = check_tensor(
      w1, config->weight_quant, {config->hidden_dim, config->dim, 0, 0}, __LINE__
    );
    _w2 = check_tensor(
      w2, config->weight_quant, {config->dim, config->hidden_dim, 0, 0}, __LINE__
    );
    _w3 = check_tensor(
      w3, config->weight_quant, {config->hidden_dim, config->dim, 0, 0}, __LINE__
    );
  }

  bool need_block_scales = _config->weight_quant == Quant::F8E5M2;
  if (need_block_scales) {
    int b0 = config->block_size[0];
    int b1 = config->block_size[1];
    if (config->q_lora_rank > 0) {
      _sq_a = static_cast<float*>(check_tensor(
        sq_a, Quant::F32, 
        {cdiv(config->q_lora_rank, b0), cdiv(config->dim, b1), 0, 0}, 
        __LINE__
      ));
      _sq_b = static_cast<float*>(check_tensor(
        sq_b, Quant::F32, 
        {cdiv(config->n_heads * config->head_dim, b0), cdiv(config->q_lora_rank, b1), 0, 0}, 
        __LINE__
      ));
    } else {
      _sq = static_cast<float*>(check_tensor(
        sq, Quant::F32, 
        {cdiv(config->n_heads * config->head_dim, b0), cdiv(config->dim, b1), 0, 0}, 
        __LINE__
      ));
    }
    _skv_a = static_cast<float*>(check_tensor(
      skv_a, Quant::F32, 
      {cdiv(config->kv_lora_rank + config->qk_rope_head_dim, b0), cdiv(config->dim, b1), 0, 0}, 
      __LINE__
    ));
    _skv_b = static_cast<float*>(check_tensor(
      skv_b, Quant::F32, 
      {cdiv(config->n_kv_heads * (config->head_dim-config->qk_rope_head_dim+config->v_head_dim), b0), cdiv(config->kv_lora_rank, b1), 0, 0}, 
      __LINE__
    ));
    _so = static_cast<float*>(check_tensor(
      so, Quant::F32, 
      {cdiv(config->dim, b0), cdiv(config->n_heads * config->v_head_dim, b1), 0, 0}, 
      __LINE__
    ));
    if (config->n_routed_experts > 0 && layer_i >= config->first_k_dense_replace) {
      _s1 = static_cast<float*>(check_tensor(
        s1, Quant::F32, 
        {config->n_routed_experts, cdiv(config->moe_intermediate_size, b0), cdiv(config->dim, b1), 0}, 
        __LINE__
      ));
      _s2 = static_cast<float*>(check_tensor(
        s2, Quant::F32, 
        {config->n_routed_experts, cdiv(config->dim, b0), cdiv(config->moe_intermediate_size, b1), 0}, 
        __LINE__
      ));
      _s3 = static_cast<float*>(check_tensor(
        s3, Quant::F32, 
        {config->n_routed_experts, cdiv(config->moe_intermediate_size, b0), cdiv(config->dim, b1), 0}, 
        __LINE__
      ));
      if (config->n_shared_experts > 0) {
        _shared_s1 = static_cast<float*>(check_tensor(
          shared_s1, Quant::F32, 
          {cdiv(config->n_shared_experts * config->moe_intermediate_size, b0), cdiv(config->dim, b1), 0}, 
          __LINE__
        ));
        _shared_s2 = static_cast<float*>(check_tensor(
          shared_s2, Quant::F32, 
          {cdiv(config->dim, b0), cdiv(config->n_shared_experts * config->moe_intermediate_size, b1), 0}, 
          __LINE__
        ));
        _shared_s3 = static_cast<float*>(check_tensor(
          shared_s3, Quant::F32, 
          {cdiv(config->n_shared_experts * config->moe_intermediate_size, b0), cdiv(config->dim, b1), 0}, 
          __LINE__
        ));
      }
    } else {
      _s1 = static_cast<float*>(check_tensor(
        s1, Quant::F32, 
        {cdiv(config->hidden_dim, b0), cdiv(config->dim, b1), 0, 0}, 
        __LINE__
      ));
      _s2 = static_cast<float*>(check_tensor(
        s2, Quant::F32, 
        {cdiv(config->dim, b0), cdiv(config->hidden_dim, b1), 0, 0}, 
        __LINE__
      ));
      _s3 = static_cast<float*>(check_tensor(
        s3, Quant::F32, 
        {cdiv(config->hidden_dim, b0), cdiv(config->dim, b1), 0, 0}, 
        __LINE__
      ));
    }
  }

  _key_cache = new f16_t[config->max_seq_len * config->n_kv_heads * config->head_dim]();
  _value_cache = new f16_t[config->max_seq_len * config->n_kv_heads * config->v_head_dim]();
}

Block::~Block() {
  if (_device == Device::CPU) {
    delete[] _key_cache;
    delete[] _value_cache;
  }
}

void Block::block(
  InferenceState& s,  // inference state
  int pos,            // index of the current token in the sequence
  int kv_sink,        // number of sink tokens currently in the KV cache
  int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
  int kv_len          // number of tokens in the kv cache that we will attend over
) const {
  if (_device == Device::CPU) {
    switch (_config->weight_quant) {
      case Quant::F32: {
        _block_cpu<float>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      }
      case Quant::F16: {
#if defined(__AVX2__) && defined(__F16C__)
        _block_cpu<f16_t>(s, pos, kv_sink, kv_pos, kv_len);
#else
        assert(false && "float16 not supported on this platform");
#endif
        break;
      }
      case Quant::F8E5M2: {
        _block_cpu<f8e5m2_t>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      }
      case Quant::Q2_K: {
        _block_cpu<block_q2_K>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      }
      default: {
        assert(false && "unsupported weight quantization for cpu");
      }
    }
  }
}

InferenceState::InferenceState(const std::shared_ptr<Config> config): 
  _config(config) {
  assert(config);
  _x = new float[config->dim]();
  _xb = new float[config->dim]();
  _xb2 = new float[std::max(config->dim, config->n_kv_heads * config->v_head_dim)]();
  _hb = new float[config->hidden_dim]();
  _hb2 = new float[config->hidden_dim]();
  if (config->q_lora_rank > 0) {
    _q_a = new float[config->q_lora_rank]();
  }
  _q = new float[config->n_heads * config->head_dim]();
  _kv_a = new float[config->kv_lora_rank + config->qk_rope_head_dim]();
  _kv_b = new float[config->n_kv_heads * (config->head_dim-config->qk_rope_head_dim+config->v_head_dim)]();
  _ropebuf = new float[config->n_kv_heads * config->qk_rope_head_dim]();
  _k = new float[config->n_kv_heads * config->head_dim]();
  _v = new float[config->n_kv_heads * config->v_head_dim]();
  _att = new float[config->n_heads * config->max_seq_len]();
  _logits = new float[config->vocab_size]();
  if (config->n_routed_experts > 0) {
    _moe_weights = new float[config->n_routed_experts]();
    _active_experts = new int[config->n_active_routed]();
    _active_experts_weights = new float[config->n_active_routed]();
  }
  _dqb = new float[config->dim * config->vocab_size]();
}

InferenceState::~InferenceState() {
  if (_device == Device::CPU) {
    delete[] _x;
    delete[] _xb;
    delete[] _xb2;
    delete[] _hb;
    delete[] _hb2;
    if (_q_a != nullptr) {
      delete[] _q_a;
    }
    delete[] _q;
    delete[] _kv_a;
    delete[] _kv_b;
    delete[] _ropebuf;
    delete[] _k;
    delete[] _v;
    delete[] _att;
    delete[] _logits;
    if (_moe_weights != nullptr) {
      delete[] _moe_weights;
      delete[] _active_experts;
      delete[] _active_experts_weights;
    }
    delete[] _dqb;
  }
}

Model::Model(YALMData& yalm, int context) {
  config = std::make_shared<Config>();
  config->from_yalm(yalm, context);
  std::cout << "loading model with quant: " << quant_to_string(config->weight_quant) << std::endl;

  bool need_weight_scales = config->weight_quant == Quant::F8E5M2;
  int b0 = config->block_size[0];
  int b1 = config->block_size[1];

  token_embedding_table = check_tensor(
    get_tensor(yalm, "model.embed.weight"), 
    config->weight_quant,
    {config->vocab_size, config->dim, 0, 0},
    __LINE__
  );
  if (need_weight_scales) {
    token_embedding_scale = static_cast<float*>(check_tensor(
      get_tensor(yalm, "model.embed.scale"), 
      Quant::F32,
      {cdiv(config->vocab_size, b0), cdiv(config->dim, b1), 0, 0},
      __LINE__
    ));
  }

  for (int i = 0; i < config->n_layers; ++i) {
    blocks.emplace_back(std::make_shared<Block>(
      i,
      config,
      get_tensor(yalm, fmt::format("model.layers.{}.attn.norm.weight", i)),
      config->q_lora_rank > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.attn.q_a_norm.weight", i)) : nullptr,
      get_tensor(yalm, fmt::format("model.layers.{}.attn.kv_a_norm.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.norm.weight", i)),
      config->q_lora_rank == 0 ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wq.weight", i)) : nullptr,
      need_weight_scales && config->q_lora_rank == 0 ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wq.scale", i)) : nullptr,
      config->q_lora_rank > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wq_a.weight", i)) : nullptr,
      need_weight_scales && config->q_lora_rank > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wq_a.scale", i)) : nullptr,
      config->q_lora_rank > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wq_b.weight", i)) : nullptr,
      need_weight_scales && config->q_lora_rank > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wq_b.scale", i)) : nullptr,
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wkv_a.weight", i)),
      need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wkv_a.scale", i)) : nullptr,
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wkv_b.weight", i)),
      need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wkv_b.scale", i)) : nullptr,
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wo.weight", i)),
      need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wo.scale", i)) : nullptr,
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w1.weight", i)),
      need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.mlp.w1.scale", i)) : nullptr,
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w2.weight", i)),
      need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.mlp.w2.scale", i)) : nullptr,
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w3.weight", i)),
      need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.mlp.w3.scale", i)) : nullptr,
      i >= config->first_k_dense_replace && config->n_shared_experts > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.shared_mlp.w1.weight", i)) : nullptr,
      need_weight_scales && i >= config->first_k_dense_replace && config->n_shared_experts > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.shared_mlp.w1.scale", i)) : nullptr,
      i >= config->first_k_dense_replace && config->n_shared_experts > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.shared_mlp.w2.weight", i)) : nullptr,
      need_weight_scales && i >= config->first_k_dense_replace && config->n_shared_experts > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.shared_mlp.w2.scale", i)) : nullptr,
      i >= config->first_k_dense_replace && config->n_shared_experts > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.shared_mlp.w3.weight", i)) : nullptr,
      need_weight_scales && i >= config->first_k_dense_replace && config->n_shared_experts > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.shared_mlp.w3.scale", i)) : nullptr,
      i >= config->first_k_dense_replace && config->n_routed_experts > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.moegate.weight", i)) : nullptr,
      i >= config->first_k_dense_replace && config->n_routed_experts > 0 && config->has_moegate_bias ? get_tensor(yalm, fmt::format("model.layers.{}.moegate.bias", i)) : nullptr
    ));
  }

  rms_final_weight = static_cast<float*>(check_tensor(
    get_tensor(yalm, "model.norm.weight"), 
    Quant::F32, 
    {config->dim, 0, 0, 0},
    __LINE__
  ));
  bool tie_word_embeddings = yalm.tensors.count("model.output.weight") == 0;
  if (tie_word_embeddings) {
    wcls = token_embedding_table;
    scls = token_embedding_scale;
  } else {
    wcls = check_tensor(
      get_tensor(yalm, "model.output.weight"), 
      config->weight_quant, 
      {config->vocab_size, config->dim, 0, 0},
      __LINE__
    );
    if (need_weight_scales) {
      scls = static_cast<float*>(check_tensor(
        get_tensor(yalm, "model.output.scale"), 
        Quant::F32, 
        {cdiv(config->vocab_size, b0), cdiv(config->dim, b1), 0, 0},
        __LINE__
      ));
    }
  }
}

void Model::forward(InferenceState& s, int token, int pos, InferenceMode mode) {
  if (s.device() != _device) {
    std::cerr << "FATAL: inference state device mismatch" << std::endl;
    assert(false);
    return;
  }
  if (_device == Device::CPU) {
    _forward_cpu(s, token, pos, mode);
  }
}

#if DEBUG_MODEL
DebugTensor::DebugTensor(const std::vector<float>& data) {
  data_f32 = data;
  data_type = DataType::F32;
}
DebugTensor::DebugTensor(const std::vector<f16_t>& data) {
  data_f16 = data;
  data_type = DataType::F16;
}

float DebugTensor::max_err(const DebugTensor& other) const {
  if (data_type != other.data_type) {
    return -1;
  }
  if (data_type == DataType::F32) {
    float max_err = 0;
    for (size_t i = 0; i < data_f32.size(); i++) {
      max_err = std::max(max_err, std::abs(data_f32[i] - other.data_f32[i]));
    }
    return max_err;
  } else {
#if defined(__F16C__)
    float max_err = 0;
    for (size_t i = 0; i < data_f16.size(); i++) {
      max_err = std::max(max_err, std::abs(_cvtsh_ss(data_f16[i]) - _cvtsh_ss(other.data_f16[i])));
    }
    return max_err;
#else
  assert(false && "float16 not supported on this platform");
#endif
  }
}
#endif