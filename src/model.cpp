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
  use_mla = yalm.metadata.contains("use_mla") ? 
    static_cast<bool>(std::stoi(yalm.metadata.at("use_mla").get<std::string>())) : false;
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
  } else if (quant == "q2_k") {
    weight_quant = Quant::Q2_K;
  } else if (quant == "q3_k") {
    weight_quant = Quant::Q3_K;
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

std::optional<QTensor> check_tensor(const Tensor* tensor, Quant weight_quant, std::array<int, 4> shape, const int debug_line) {
  if (tensor == nullptr) {
    std::cerr << "FATAL: missing tensor at line " << debug_line << std::endl;
    assert(false);
    return std::nullopt;
  }
  return QTensor::from_codec_tensor(*tensor, weight_quant, shape, debug_line);
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
  const Tensor* rms_ffn_weight,
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
) : _layer_i(layer_i), _config(config) {
  switch (config->weight_quant) {
    case Quant::F32:
    case Quant::F16:
    case Quant::F8E5M2:
    case Quant::Q2_K:
    case Quant::Q3_K: {
      break;
    }
    default: {
      std::cerr << "FATAL: unsupported weight quantization: " << quant_to_string(config->weight_quant) << std::endl;
      assert(false);
      break;
    }
  }

  _rms_att_weight = check_tensor(
    rms_att_weight, Quant::F32, {config->dim, 0, 0, 0}, __LINE__
  );
  _rms_ffn_weight = check_tensor(
    rms_ffn_weight, Quant::F32, {config->dim, 0, 0, 0}, __LINE__
  );

  bool need_block_scales = _config->weight_quant == Quant::F8E5M2;
  int b0 = config->block_size[0];
  int b1 = config->block_size[1];

  if (config->n_routed_experts > 0 && layer_i >= config->first_k_dense_replace) {
    _moegate = check_tensor(
      moegate, Quant::F32, {config->n_routed_experts, config->dim, 0, 0}, __LINE__
    );
    if (moegate_bias != nullptr) {
      _moegate_bias = check_tensor(
        moegate_bias, Quant::F32, {config->n_routed_experts, 0, 0, 0}, __LINE__
      );
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
    if (need_block_scales) {
      _s1 = check_tensor(
        s1, Quant::F32,
        {config->n_routed_experts, cdiv(config->moe_intermediate_size, b0), cdiv(config->dim, b1), 0},
        __LINE__
      );
      _s2 = check_tensor(
        s2, Quant::F32,
        {config->n_routed_experts, cdiv(config->dim, b0), cdiv(config->moe_intermediate_size, b1), 0},
        __LINE__
      );
      _s3 = check_tensor(
        s3, Quant::F32,
        {config->n_routed_experts, cdiv(config->moe_intermediate_size, b0), cdiv(config->dim, b1), 0},
        __LINE__
      );
    }
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
      if (need_block_scales) {
        _shared_s1 = check_tensor(
          shared_s1, Quant::F32,
          {cdiv(config->n_shared_experts * config->moe_intermediate_size, b0), cdiv(config->dim, b1), 0},
          __LINE__
        );
        _shared_s2 = check_tensor(
          shared_s2, Quant::F32,
          {cdiv(config->dim, b0), cdiv(config->n_shared_experts * config->moe_intermediate_size, b1), 0},
          __LINE__
        );
        _shared_s3 = check_tensor(
          shared_s3, Quant::F32,
          {cdiv(config->n_shared_experts * config->moe_intermediate_size, b0), cdiv(config->dim, b1), 0},
          __LINE__
        );
      }
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
    if (need_block_scales) {
      _s1 = check_tensor(
        s1, Quant::F32,
        {cdiv(config->hidden_dim, b0), cdiv(config->dim, b1), 0, 0},
        __LINE__
      );
      _s2 = check_tensor(
        s2, Quant::F32,
        {cdiv(config->dim, b0), cdiv(config->hidden_dim, b1), 0, 0},
        __LINE__
      );
      _s3 = check_tensor(
        s3, Quant::F32,
        {cdiv(config->hidden_dim, b0), cdiv(config->dim, b1), 0, 0},
        __LINE__
      );
    }
  }
}

Block::~Block() {}

void Block::block(
  InferenceState& s,
  int pos,
  int kv_sink,
  int kv_pos,
  int kv_len
) const {
  if (_device == Device::CPU) {
    switch (_config->weight_quant) {
      case Quant::F32:
        _block_cpu<float>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      case Quant::F16:
#if defined(__AVX2__) && defined(__F16C__)
        _block_cpu<f16_t>(s, pos, kv_sink, kv_pos, kv_len);
#else
        assert(false && "float16 not supported on this platform");
#endif
        break;
      case Quant::F8E5M2:
        _block_cpu<f8e5m2_t>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      case Quant::Q2_K:
        _block_cpu<block_q2_K>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      case Quant::Q3_K:
        _block_cpu<block_q3_K>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      default:
        assert(false && "unsupported weight quantization for cpu");
    }
  }
}

double Block::active_bytes(size_t pos) const {
  double bytes_per_weight = bits_per_weight(_config->weight_quant, _config->block_size[0] * _config->block_size[1]) / 8.0;

  double bytes = 0;
  bytes += _rms_att_weight->size;
  bytes += _rms_ffn_weight->size;
  if (_config->n_routed_experts > 0 && _w1->ndim() == 3) {
    bytes += _moegate->size; // moegate
    bytes += _moegate_bias->size; // moegate_bias
    // bytes_per_weight accounts for scales and other quantization schemes
    bytes += _config->n_active_routed * 3 * _config->dim * _config->moe_intermediate_size * bytes_per_weight; // w1, w2, w3
  } else {
    bytes += _w1->size + _w2->size + _w3->size; // w1, w2, w3
    if (_s1) {
      bytes += _s1->size;
      bytes += _s2->size;
      bytes += _s3->size;
    }
  }
  if (_config->n_shared_experts > 0) {
    bytes += _shared_w1->size + _shared_w2->size + _shared_w3->size; // shared_w1, shared_w2, shared_w3
    if (_shared_s1) {
      bytes += _shared_s1->size;
      bytes += _shared_s2->size;
      bytes += _shared_s3->size;
    }
  }
  return bytes;
}

BlockMHA::BlockMHA(
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
  const Tensor* wkv_a,
  const Tensor* skv_a,
  const Tensor* wq_b,
  const Tensor* sq_b,
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
) : Block(layer_i, config, rms_att_weight, rms_ffn_weight, w1, s1, w2, s2, w3, s3, shared_w1, shared_s1, shared_w2, shared_s2, shared_w3, shared_s3, moegate, moegate_bias) {

  bool need_block_scales = _config->weight_quant == Quant::F8E5M2;
  int b0 = config->block_size[0];
  int b1 = config->block_size[1];

  if (config->q_lora_rank > 0) {
    _rms_q_a_weight = check_tensor(
      rms_q_a_weight, Quant::F32, {config->q_lora_rank, 0, 0, 0}, __LINE__
    );
    _wq_a = check_tensor(
      wq_a, config->weight_quant, {config->q_lora_rank, config->dim, 0, 0}, __LINE__
    );
    _wq_b = check_tensor(
      wq_b, config->weight_quant, {config->n_heads * config->head_dim, config->q_lora_rank, 0, 0}, __LINE__
    );
    if (need_block_scales) {
      _sq_a = check_tensor(
        sq_a, Quant::F32,
        {cdiv(config->q_lora_rank, b0), cdiv(config->dim, b1), 0, 0},
        __LINE__
      );
      _sq_b = check_tensor(
        sq_b, Quant::F32,
        {cdiv(config->n_heads * config->head_dim, b0), cdiv(config->q_lora_rank, b1), 0, 0},
        __LINE__
      );
    }
  } else {
    _wq = check_tensor(
      wq, config->weight_quant, {config->n_heads * config->head_dim, config->dim, 0, 0}, __LINE__
    );
    if (need_block_scales) {
      _sq = check_tensor(
        sq, Quant::F32,
        {cdiv(config->n_heads * config->head_dim, b0), cdiv(config->dim, b1), 0, 0},
        __LINE__
      );
    }
  }

  _rms_kv_a_weight = check_tensor(
    rms_kv_a_weight, Quant::F32, {config->kv_lora_rank, 0, 0, 0}, __LINE__ // Assuming kv_lora_rank is correct size here for MHA norm too
  );
  _wkv_a = check_tensor(
    wkv_a, config->weight_quant, {config->kv_lora_rank + config->qk_rope_head_dim, config->dim, 0, 0}, __LINE__
  );
  _wkv_b = check_tensor(
    wkv_b, config->weight_quant, {config->n_heads * (config->head_dim-config->qk_rope_head_dim+config->v_head_dim), config->kv_lora_rank, 0, 0}, __LINE__
  );
  _wo = check_tensor(
    wo, config->weight_quant, {config->dim, config->n_heads * config->v_head_dim, 0, 0}, __LINE__
  );

  if (need_block_scales) {
    _skv_a = check_tensor(
      skv_a, Quant::F32,
      {cdiv(config->kv_lora_rank + config->qk_rope_head_dim, b0), cdiv(config->dim, b1), 0, 0},
      __LINE__
    );
    _skv_b = check_tensor(
      skv_b, Quant::F32,
      {cdiv(config->n_heads * (config->head_dim-config->qk_rope_head_dim+config->v_head_dim), b0), cdiv(config->kv_lora_rank, b1), 0, 0},
      __LINE__
    );
    _so = check_tensor(
      so, Quant::F32,
      {cdiv(config->dim, b0), cdiv(config->n_heads * config->v_head_dim, b1), 0, 0},
      __LINE__
    );
  }

  _key_cache = new f16_t[config->max_seq_len * config->n_heads * config->head_dim]();
  _value_cache = new f16_t[config->max_seq_len * config->n_heads * config->v_head_dim]();
}

BlockMHA::~BlockMHA() {
  if (_device == Device::CPU) {
    delete[] _key_cache;
    delete[] _value_cache;
  }
}

double BlockMHA::active_bytes(size_t pos) const {
  double bytes = Block::active_bytes(pos);
  
  // Add active bytes for attention and KV cache
  if (_wq) bytes += _wq->size;
  if (_sq) bytes += _sq->size;
  if (_wq_a) bytes += _wq_a->size;
  if (_sq_a) bytes += _sq_a->size;
  if (_wkv_a) bytes += _wkv_a->size;
  if (_skv_a) bytes += _skv_a->size;
  if (_wo) bytes += _wo->size;
  if (_so) bytes += _so->size;
  if (_wq_b) bytes += _wq_b->size;
  if (_sq_b) bytes += _sq_b->size;
  if (_wkv_b) bytes += _wkv_b->size;
  if (_skv_b) bytes += _skv_b->size;
  
  size_t kv_len = std::min(static_cast<size_t>(_config->max_seq_len), pos + 1);
  size_t kv_entry_size = sizeof(f16_t);
  bytes += 2 * kv_len * _config->n_heads * _config->head_dim * kv_entry_size; // key_cache, value_cache
  return bytes;
}

void BlockMHA::attention_impl(
  InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len
) const {
  switch (_config->weight_quant) {
    case Quant::F32:
      _attention_impl<float>(s, pos, kv_sink, kv_pos, kv_len);
      break;
    case Quant::F16:
      _attention_impl<f16_t>(s, pos, kv_sink, kv_pos, kv_len);
      break;
    case Quant::F8E5M2:
      _attention_impl<f8e5m2_t>(s, pos, kv_sink, kv_pos, kv_len);
      break;
    case Quant::Q2_K:
      _attention_impl<block_q2_K>(s, pos, kv_sink, kv_pos, kv_len);
      break;
    case Quant::Q3_K:
      _attention_impl<block_q3_K>(s, pos, kv_sink, kv_pos, kv_len);
      break;
    default:
      assert(false && "unsupported weight quantization for mha");
  }
}


BlockMLA::BlockMLA(
  int layer_i,
  const std::shared_ptr<Config> config,
  const Tensor* rms_att_weight,
  const Tensor* rms_q_a_weight,
  const Tensor* rms_kv_a_weight,
  const Tensor* rms_ffn_weight,
  const Tensor* wq_a,
  const Tensor* sq_a,
  const Tensor* wkv_a,
  const Tensor* skv_a,
  const Tensor* wo,
  const Tensor* so,
  const Tensor* wc,
  const Tensor* sc,
  const Tensor* wq_rope_b,
  const Tensor* sq_rope_b,
  const Tensor* wv_b,
  const Tensor* sv_b,
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
) : Block(layer_i, config, rms_att_weight, rms_ffn_weight, w1, s1, w2, s2, w3, s3, shared_w1, shared_s1, shared_w2, shared_s2, shared_w3, shared_s3, moegate, moegate_bias) {

  bool need_block_scales = _config->weight_quant == Quant::F8E5M2;
  int b0 = config->block_size[0];
  int b1 = config->block_size[1];

  _rms_q_a_weight = check_tensor(
    rms_q_a_weight, Quant::F32, {config->q_lora_rank, 0, 0, 0}, __LINE__
  );
  _rms_kv_a_weight = check_tensor(
    rms_kv_a_weight, Quant::F32, {config->kv_lora_rank, 0, 0, 0}, __LINE__ // Only norm latent part
  );

  _wq_a = check_tensor(
    wq_a, config->weight_quant, {config->q_lora_rank, config->dim, 0, 0}, __LINE__
  );
  _wkv_a = check_tensor(
    wkv_a, config->weight_quant, {config->kv_lora_rank + config->qk_rope_head_dim, config->dim, 0, 0}, __LINE__
  );
  _wc = check_tensor(
    wc, config->weight_quant, {config->n_heads * config->kv_lora_rank, config->q_lora_rank, 0, 0}, __LINE__
  );
  _wq_rope_b = check_tensor(
    wq_rope_b, config->weight_quant, {config->n_heads * config->qk_rope_head_dim, config->q_lora_rank, 0, 0}, __LINE__
  );
  _wv_b = check_tensor(
      wv_b, config->weight_quant, {config->n_heads * config->v_head_dim, config->kv_lora_rank, 0, 0}, __LINE__
  );
  // Reshape _wv_b from 2D to 3D
  _wv_b = QTensor(_wv_b->quant, {config->n_heads, config->v_head_dim, config->kv_lora_rank}, _wv_b->data, _wv_b->size);
  _wo = check_tensor(
    wo, config->weight_quant, {config->dim, config->n_heads * config->v_head_dim, 0, 0}, __LINE__
  );

  if (need_block_scales) {
    _sq_a = check_tensor(
      sq_a, Quant::F32,
      {cdiv(config->q_lora_rank, b0), cdiv(config->dim, b1), 0, 0},
      __LINE__
    );
    _skv_a = check_tensor(
      skv_a, Quant::F32,
      {cdiv(config->kv_lora_rank + config->qk_rope_head_dim, b0), cdiv(config->dim, b1), 0, 0},
      __LINE__
    );
    _sc = check_tensor(
      sc, Quant::F32,
      {cdiv(config->n_heads * config->kv_lora_rank, b0), cdiv(config->q_lora_rank, b1), 0, 0},
      __LINE__
    );
    _sq_rope_b = check_tensor(
      sq_rope_b, Quant::F32,
      {cdiv(config->n_heads * config->qk_rope_head_dim, b0), cdiv(config->q_lora_rank, b1), 0, 0},
      __LINE__
    );
    _sv_b = check_tensor(
      sv_b, Quant::F32,
      {cdiv(config->n_heads * config->v_head_dim, b0), cdiv(config->kv_lora_rank, b1), 0, 0},
      __LINE__
    );
    _so = check_tensor(
      so, Quant::F32,
      {cdiv(config->dim, b0), cdiv(config->n_heads * config->v_head_dim, b1), 0, 0},
      __LINE__
    );
  }

  _kv_nope_cache = new f16_t[config->max_seq_len * config->kv_lora_rank]();
  _kv_rope_cache = new f16_t[config->max_seq_len * config->qk_rope_head_dim]();
}

BlockMLA::~BlockMLA() {
  if (_device == Device::CPU) {
    delete[] _kv_nope_cache;
    delete[] _kv_rope_cache;
  }
}

double BlockMLA::active_bytes(size_t pos) const {
  double bytes = Block::active_bytes(pos);

  bytes += _rms_q_a_weight->size;
  bytes += _rms_kv_a_weight->size;
  if (_wq_a) bytes += _wq_a->size;
  if (_sq_a) bytes += _sq_a->size;
  if (_wkv_a) bytes += _wkv_a->size;
  if (_skv_a) bytes += _skv_a->size;
  if (_wo) bytes += _wo->size;
  if (_so) bytes += _so->size;
  if (_wc) bytes += _wc->size;
  if (_sc) bytes += _sc->size;
  if (_wq_rope_b) bytes += _wq_rope_b->size;
  if (_sq_rope_b) bytes += _sq_rope_b->size;
  if (_wv_b) bytes += _wv_b->size;
  if (_sv_b) bytes += _sv_b->size;
  size_t kv_len = std::min(static_cast<size_t>(_config->max_seq_len), pos + 1);
  size_t kv_entry_size = sizeof(f16_t);
  bytes += kv_len * _config->kv_lora_rank * kv_entry_size; // kv_nope_cache
  bytes += kv_len * _config->qk_rope_head_dim * kv_entry_size; // kv_rope_cache
  return bytes;
}

void BlockMLA::attention_impl(
  InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len
) const {
  switch (_config->weight_quant) {
    case Quant::F32:
      _attention_impl<float>(s, pos, kv_sink, kv_pos, kv_len);
      break;
    case Quant::F16:
      _attention_impl<f16_t>(s, pos, kv_sink, kv_pos, kv_len);
      break;
    case Quant::F8E5M2:
      _attention_impl<f8e5m2_t>(s, pos, kv_sink, kv_pos, kv_len);
      break;
    case Quant::Q2_K:
      _attention_impl<block_q2_K>(s, pos, kv_sink, kv_pos, kv_len);
      break;
    case Quant::Q3_K:
      _attention_impl<block_q3_K>(s, pos, kv_sink, kv_pos, kv_len);
      break;
    default:
      assert(false && "unsupported weight quantization for mla");
  }
}

InferenceState::InferenceState(const std::shared_ptr<Config> config): 
  _config(config) {
  assert(config);
  _x = new float[config->dim]();
  _xb = new float[config->dim]();
  _xb2 = new float[std::max({
    config->dim, 
    config->n_heads * config->v_head_dim, 
    config->n_heads * config->kv_lora_rank
  })]();
  _hb = new float[std::max({
    config->dim, 
    config->hidden_dim
  })]();
  _hb2 = new float[config->hidden_dim]();
  if (config->q_lora_rank > 0) {
    _q_a = new float[config->q_lora_rank]();
  }
  _q = new float[config->n_heads * config->head_dim]();
  _kv_a = new float[config->kv_lora_rank + config->qk_rope_head_dim]();
  _kv_b = new float[config->n_heads * (config->head_dim-config->qk_rope_head_dim+config->v_head_dim)]();
  _ropebuf = new float[config->n_heads * config->qk_rope_head_dim]();
  _k = new float[config->n_heads * config->head_dim]();
  _v = new float[config->n_heads * config->v_head_dim]();
  _att = new float[config->n_heads * config->max_seq_len]();
  _logits = new float[config->vocab_size]();
  if (config->use_mla) {
    _q_c = new float[config->n_heads * config->kv_lora_rank]();
    _q_rope = new float[config->n_heads * config->qk_rope_head_dim]();
  }
  if (config->n_routed_experts > 0) {
    _moe_weights = new float[config->n_routed_experts]();
    _active_experts = new int[config->n_active_routed]();
    _active_experts_weights = new float[config->n_active_routed]();
  }
  // TODO: consider dynamically resizing based on inputs
  size_t aqb_nitems = std::max({
    config->dim,
    config->moe_intermediate_size,
    config->n_heads * config->v_head_dim,
    config->n_heads * config->kv_lora_rank,
    config->hidden_dim
  });
  size_t aqb_nblocks = aqb_nitems / QK_K;
  _aqb = new uint8_t[aqb_nblocks * sizeof(block_q8_K)]();
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
    delete[] _aqb;
  }
}

Model::Model(YALMData& yalm, int context) {
  config = std::make_shared<Config>();
  config->from_yalm(yalm, context);
  std::cout << "loading model with quant: " << quant_to_string(config->weight_quant) << std::endl;

  bool need_weight_scales = config->weight_quant == Quant::F8E5M2;
  bool use_mla = config->use_mla;
  int b0 = config->block_size[0];
  int b1 = config->block_size[1];

  token_embedding_table = check_tensor(
    get_tensor(yalm, "model.embed.weight"),
    config->weight_quant,
    {config->vocab_size, config->dim, 0, 0},
    __LINE__
  );
  if (need_weight_scales) {
    token_embedding_scale = check_tensor(
      get_tensor(yalm, "model.embed.scale"),
      Quant::F32,
      {cdiv(config->vocab_size, b0), cdiv(config->dim, b1), 0, 0},
      __LINE__
    );
  }

  for (int i = 0; i < config->n_layers; ++i) {
    const Tensor* p_rms_att_weight = get_tensor(yalm, fmt::format("model.layers.{}.attn.norm.weight", i));
    const Tensor* p_rms_ffn_weight = get_tensor(yalm, fmt::format("model.layers.{}.mlp.norm.weight", i));
    const Tensor* p_w1 = get_tensor(yalm, fmt::format("model.layers.{}.mlp.w1.weight", i));
    const Tensor* p_s1 = need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.mlp.w1.scale", i)) : nullptr;
    const Tensor* p_w2 = get_tensor(yalm, fmt::format("model.layers.{}.mlp.w2.weight", i));
    const Tensor* p_s2 = need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.mlp.w2.scale", i)) : nullptr;
    const Tensor* p_w3 = get_tensor(yalm, fmt::format("model.layers.{}.mlp.w3.weight", i));
    const Tensor* p_s3 = need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.mlp.w3.scale", i)) : nullptr;
    const Tensor* p_shared_w1 = (i >= config->first_k_dense_replace && config->n_shared_experts > 0) ? get_tensor(yalm, fmt::format("model.layers.{}.shared_mlp.w1.weight", i)) : nullptr;
    const Tensor* p_shared_s1 = (need_weight_scales && i >= config->first_k_dense_replace && config->n_shared_experts > 0) ? get_tensor(yalm, fmt::format("model.layers.{}.shared_mlp.w1.scale", i)) : nullptr;
    const Tensor* p_shared_w2 = (i >= config->first_k_dense_replace && config->n_shared_experts > 0) ? get_tensor(yalm, fmt::format("model.layers.{}.shared_mlp.w2.weight", i)) : nullptr;
    const Tensor* p_shared_s2 = (need_weight_scales && i >= config->first_k_dense_replace && config->n_shared_experts > 0) ? get_tensor(yalm, fmt::format("model.layers.{}.shared_mlp.w2.scale", i)) : nullptr;
    const Tensor* p_shared_w3 = (i >= config->first_k_dense_replace && config->n_shared_experts > 0) ? get_tensor(yalm, fmt::format("model.layers.{}.shared_mlp.w3.weight", i)) : nullptr;
    const Tensor* p_shared_s3 = (need_weight_scales && i >= config->first_k_dense_replace && config->n_shared_experts > 0) ? get_tensor(yalm, fmt::format("model.layers.{}.shared_mlp.w3.scale", i)) : nullptr;
    const Tensor* p_moegate = (i >= config->first_k_dense_replace && config->n_routed_experts > 0) ? get_tensor(yalm, fmt::format("model.layers.{}.moegate.weight", i)) : nullptr;
    const Tensor* p_moegate_bias = (i >= config->first_k_dense_replace && config->n_routed_experts > 0 && config->has_moegate_bias) ? get_tensor(yalm, fmt::format("model.layers.{}.moegate.bias", i)) : nullptr;

    const Tensor* p_rms_q_a_weight = config->q_lora_rank > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.attn.q_a_norm.weight", i)) : nullptr;
    const Tensor* p_rms_kv_a_weight = get_tensor(yalm, fmt::format("model.layers.{}.attn.kv_a_norm.weight", i));
    const Tensor* p_wq_a = config->q_lora_rank > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wq_a.weight", i)) : nullptr;
    const Tensor* p_sq_a = (need_weight_scales && config->q_lora_rank > 0) ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wq_a.scale", i)) : nullptr;
    const Tensor* p_wkv_a = get_tensor(yalm, fmt::format("model.layers.{}.attn.wkv_a.weight", i));
    const Tensor* p_skv_a = need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wkv_a.scale", i)) : nullptr;
    const Tensor* p_wo = get_tensor(yalm, fmt::format("model.layers.{}.attn.wo.weight", i));
    const Tensor* p_so = need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wo.scale", i)) : nullptr;


    if (use_mla) {
      const Tensor* p_wc = get_tensor(yalm, fmt::format("model.layers.{}.attn.wc.weight", i));
      const Tensor* p_sc = need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wc.scale", i)) : nullptr;
      const Tensor* p_wq_rope_b = get_tensor(yalm, fmt::format("model.layers.{}.attn.wq_rope_b.weight", i));
      const Tensor* p_sq_rope_b = need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wq_rope_b.scale", i)) : nullptr;
      const Tensor* p_wv_b = get_tensor(yalm, fmt::format("model.layers.{}.attn.wv_b.weight", i));
      const Tensor* p_sv_b = need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wv_b.scale", i)) : nullptr;

      blocks.emplace_back(std::make_shared<BlockMLA>(
        i, config,
        p_rms_att_weight, p_rms_q_a_weight, p_rms_kv_a_weight, p_rms_ffn_weight,
        p_wq_a, p_sq_a, p_wkv_a, p_skv_a, p_wo, p_so,
        p_wc, p_sc, p_wq_rope_b, p_sq_rope_b, p_wv_b, p_sv_b,
        p_w1, p_s1, p_w2, p_s2, p_w3, p_s3,
        p_shared_w1, p_shared_s1, p_shared_w2, p_shared_s2, p_shared_w3, p_shared_s3,
        p_moegate, p_moegate_bias
      ));
    } else {
      const Tensor* p_wq = config->q_lora_rank == 0 ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wq.weight", i)) : nullptr;
      const Tensor* p_sq = (need_weight_scales && config->q_lora_rank == 0) ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wq.scale", i)) : nullptr;
      const Tensor* p_wq_b = config->q_lora_rank > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wq_b.weight", i)) : nullptr;
      const Tensor* p_sq_b = (need_weight_scales && config->q_lora_rank > 0) ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wq_b.scale", i)) : nullptr;
      const Tensor* p_wkv_b = get_tensor(yalm, fmt::format("model.layers.{}.attn.wkv_b.weight", i));
      const Tensor* p_skv_b = need_weight_scales ? get_tensor(yalm, fmt::format("model.layers.{}.attn.wkv_b.scale", i)) : nullptr;

      blocks.emplace_back(std::make_shared<BlockMHA>(
        i, config,
        p_rms_att_weight, p_rms_q_a_weight, p_rms_kv_a_weight, p_rms_ffn_weight,
        p_wq, p_sq, p_wq_a, p_sq_a, p_wkv_a, p_skv_a,
        p_wq_b, p_sq_b, p_wkv_b, p_skv_b, p_wo, p_so,
        p_w1, p_s1, p_w2, p_s2, p_w3, p_s3,
        p_shared_w1, p_shared_s1, p_shared_w2, p_shared_s2, p_shared_w3, p_shared_s3,
        p_moegate, p_moegate_bias
      ));
    }
  }

  rms_final_weight = check_tensor(
    get_tensor(yalm, "model.norm.weight"),
    Quant::F32,
    {config->dim, 0, 0, 0},
    __LINE__
  );
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
      scls = check_tensor(
        get_tensor(yalm, "model.output.scale"),
        Quant::F32,
        {cdiv(config->vocab_size, b0), cdiv(config->dim, b1), 0, 0},
        __LINE__
      );
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

double Model::active_bytes(size_t pos) const {
  double bytes_per_weight = bits_per_weight(config->weight_quant, config->block_size[0] * config->block_size[1]) / 8.0;

  double bytes = 0;
  bytes += config->dim * bytes_per_weight; // 1 row of token_embedding_table
  // blocks
  for (auto& block : blocks) {
    bytes += block->active_bytes(pos);
  }
  bytes += rms_final_weight->size;
  bytes += wcls->size;
  if (scls) {
    bytes += scls->size;
  }

  return bytes;
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