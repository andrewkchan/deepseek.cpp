# Converts a model consisting of a huggingface config.json, tokenizer.json, and .safetensors weights into a .yalm file,
# which:
# - Normalizes the config to a common format in the header
# - Combines any safetensors shards
# - Reads the token vocabulary into a simpler format
# - Performs quantization if specified

import argparse
import os
import json
import safetensors
from safetensors.torch import save_file
import torch

from quantizer import quantize_q2_k

from typing import Tuple, List, Literal
import dataclasses

SUPPORTED_ARCHITECTURES = [
  "DeepseekV2ForCausalLM",
  "DeepseekV3ForCausalLM",
]

@dataclasses.dataclass
class BlockQuant:
  name: Literal["fp32", "fp16", "f8e5m2"]
  block_size: Tuple[int, int] | None
  dtype: torch.dtype

@dataclasses.dataclass
class KQuant:
  name: Literal["q2_k"]
  dtype: torch.dtype

Quant = BlockQuant | KQuant

SUPPORTED_QUANTS = {
  "fp32": BlockQuant(name="fp32", block_size=None, dtype=torch.float32),
  "fp16": BlockQuant(name="fp16", block_size=None, dtype=torch.float16),
  "f8e5m2": BlockQuant(name="f8e5m2", block_size=(128, 128), dtype=torch.float8_e5m2),
  "q2_k": KQuant(name="q2_k", dtype=torch.uint8),
}

class Metadata:
  def __init__(self, config, quant, n_layers, use_mla):
    arch = config["architectures"][0]
    if arch not in SUPPORTED_ARCHITECTURES:
      raise Exception(f"Architecture {arch} is not supported, must be one of {SUPPORTED_ARCHITECTURES}")
    self.arch = arch
    self.use_mla = bool(use_mla)
    if quant not in SUPPORTED_QUANTS:
      raise Exception(f"Quantization {quant} is not supported, must be one of {SUPPORTED_QUANTS}")
    self.quant: Quant = SUPPORTED_QUANTS[quant]
    if arch in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]:
      self.dim = config["hidden_size"]
      self.hidden_dim = config["intermediate_size"]
      self.n_layers = config["num_hidden_layers"]
      if n_layers is not None and self.n_layers > n_layers:
        self.n_layers = n_layers
      self.n_heads = config["num_attention_heads"]
      self.vocab_size = config["vocab_size"]
      self.max_seq_len = config["max_position_embeddings"]
      self.bos_token_id = config["bos_token_id"]
      self.eos_token_id = config["eos_token_id"]
      self.rope_theta = config.get("rope_theta", 10000.0)
      self.norm_eps = config["rms_norm_eps"]
      self.norm_type = "rmsnorm"
      
      # quantization
      self.original_quantization_config = config.get("quantization_config", None)
      if self.original_quantization_config is not None:
        dequant_block_sizes = self.original_quantization_config["weight_block_size"]
        assert type(dequant_block_sizes) == list and len(dequant_block_sizes) == 2
        assert self.original_quantization_config["quant_method"] == "fp8"

      assert config.get("attention_bias", False) == False
      assert config.get("mlp_bias", False) == False

      assert config["hidden_act"] in ["gelu", "silu"]
      self.act_type = config["hidden_act"]
      self.first_k_dense_replace = config["first_k_dense_replace"]

      # multi-latent attention
      self.kv_lora_rank = config["kv_lora_rank"]
      self.q_lora_rank = config["q_lora_rank"] or 0
      if self.use_mla:
        # TODO: support MLA with q_lora_rank == 0 (DeepSeek V2 Lite)
        assert self.q_lora_rank > 0 and self.kv_lora_rank > 0
      self.qk_nope_head_dim = config["qk_nope_head_dim"]
      self.qk_rope_head_dim = config["qk_rope_head_dim"]
      self.v_head_dim = config["v_head_dim"]

      # mixture of experts
      self.n_shared_experts = config["n_shared_experts"]
      self.n_routed_experts = config["n_routed_experts"]
      self.n_active_routed = config["num_experts_per_tok"]
      self.moe_intermediate_size = config["moe_intermediate_size"]
      self.routed_scaling_factor = config["routed_scaling_factor"]
      self.n_group = config["n_group"]
      self.norm_topk_prob = config["norm_topk_prob"]
      self.scoring_func = config["scoring_func"]
      self.topk_group = config["topk_group"]
      self.topk_method = config["topk_method"]
      if self.topk_method == "noaux_tc":
        self.topk_method = "group_limited_greedy" # TODO: support for Deepseek v3
  
  def to_dict(self):
    result = {}
    result["arch"] = self.arch
    result["use_mla"] = str(int(self.use_mla))
    result["quant"] = self.quant.name
    if self.arch in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]:
      result["dim"] = str(self.dim)
      result["hidden_dim"] = str(self.hidden_dim)
      result["n_layers"] = str(self.n_layers)
      result["n_heads"] = str(self.n_heads)
      result["vocab_size"] = str(self.vocab_size)
      result["max_seq_len"] = str(self.max_seq_len)
      result["bos_token_id"] = str(self.bos_token_id)
      result["eos_token_id"] = str(self.eos_token_id)
      result["rope_theta"] = str(self.rope_theta)
      result["norm_eps"] = str(self.norm_eps)
      result["norm_type"] = str(self.norm_type)
      result["act_type"] = str(self.act_type)
      result["first_k_dense_replace"] = str(self.first_k_dense_replace)
      # quantization
      if isinstance(self.quant, BlockQuant) and self.quant.block_size is not None:
        result["quantization_block_size_0"] = str(self.quant.block_size[0])
        result["quantization_block_size_1"] = str(self.quant.block_size[1])
      # multi-latent attention
      result["kv_lora_rank"] = str(self.kv_lora_rank)
      result["q_lora_rank"] = str(self.q_lora_rank)
      result["qk_nope_head_dim"] = str(self.qk_nope_head_dim)
      result["qk_rope_head_dim"] = str(self.qk_rope_head_dim)
      result["v_head_dim"] = str(self.v_head_dim)
      # mixture of experts
      result["n_shared_experts"] = str(self.n_shared_experts)
      result["n_routed_experts"] = str(self.n_routed_experts)
      result["n_active_routed"] = str(self.n_active_routed)
      result["moe_intermediate_size"] = str(self.moe_intermediate_size)
      result["routed_scaling_factor"] = str(self.routed_scaling_factor)
      result["n_group"] = str(self.n_group)
      result["norm_topk_prob"] = str(self.norm_topk_prob)
      result["scoring_func"] = str(self.scoring_func)
      result["topk_group"] = str(self.topk_group)
      result["topk_method"] = str(self.topk_method)
    return result

# this is a horrible gpt-2 unicode byte encoder hack from https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
# this has poisoned all HF tokenizer configs that use ByteLevel decoder/preprocessor
# as a result we get crazy UTF-8-as-bytes-as-UTF8 in the tokenizer data that we need to convert back
def gpt2_bytes_to_unicode():
  bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
  cs = bs[:]
  n = 0
  for b in range(2**8):
    if b not in bs:
      bs.append(b)
      cs.append(2**8+n)
      n += 1
  cs = [chr(n) for n in cs]
  return dict(zip(bs, cs))

def load_tokens(tokenizer_path, vocab_size):
  tokens = [""] * vocab_size
  with open(tokenizer_path, "r") as f:
    tokenizer = json.load(f)
  use_gpt2_byte_preprocessing = not tokenizer["model"].get("byte_fallback", False)
  
  vocab = tokenizer["model"]["vocab"]
  assert len(vocab) <= vocab_size

  for t, i in vocab.items():
    tokens[i] = t
  
  for added in tokenizer["added_tokens"]:
    tokens[added["id"]] = added["content"]
  
  gpt2_decode = {v: k for k, v in gpt2_bytes_to_unicode().items()}
  # Preprocess tokens into UTF-8 encoding
  for i, t in enumerate(tokens):
    if use_gpt2_byte_preprocessing:
      b = bytes([gpt2_decode.get(c, 0) for c in t])
    else:
      t = t.replace('\u2581', ' ') # sentencepiece uses this character as whitespace
      b = t.encode('utf-8')
    b = b.replace(b"\0", b"\7") # replace null bytes with bell characters
    assert b.count(0) == 0 # no null bytes allowed
    tokens[i] = b
  
  return tokens

def per_tensor_quantize(tensor: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
  """Quantize a tensor using per-tensor static scaling factor.
  Args:
      tensor: The input tensor.
      dtype: The data type to quantize to.
  """
  finfo = torch.finfo(dtype)
  # Calculate the scale as dtype max divided by absmax.
  # Since .abs() creates a new tensor, we use aminmax to get
  # the min and max first and then calculate the absmax.
  if tensor.numel() == 0:
    # Deal with empty tensors (triggered by empty MoE experts)
    min_val, max_val = (
      torch.tensor(-16.0, dtype=tensor.dtype),
      torch.tensor(16.0, dtype=tensor.dtype),
    )
  else:
    min_val, max_val = tensor.aminmax()
  amax = torch.maximum(min_val.abs(), max_val.abs())
  scale = finfo.max / amax.clamp(min=1e-12)
  # scale and clamp the tensor to bring it to
  # the representative range of float8 data type
  # (as default cast is unsaturated)
  qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
  # Return both float8 data and the inverse scale (as float),
  # as both required as inputs to torch._scaled_mm
  qweight = qweight.to(dtype)
  scale = scale.float().reciprocal()
  return qweight, scale

def per_tensor_dequantize(qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  assert scale.numel() == 1
  return qweight.to(torch.float32) * scale

def blockwise_dequantize(qweight: torch.Tensor, scale: torch.Tensor, block_size: torch.Tensor) -> torch.Tensor:
  assert qweight.ndim == scale.ndim and scale.ndim == block_size.numel() and scale.ndim == 2
  assert torch.all((torch.tensor(list(qweight.shape)) / block_size).ceil() == torch.tensor(list(scale.shape)))
  out = torch.empty_like(qweight, dtype=torch.float32)
  for i in range(scale.shape[0]):
    for j in range(scale.shape[1]):
      block_size_i = block_size[0]
      block_size_j = block_size[1]
      qw_block = qweight[i*block_size_i:(i+1)*block_size_i, j*block_size_j:(j+1)*block_size_j]
      out[i*block_size_i:(i+1)*block_size_i, j*block_size_j:(j+1)*block_size_j] = per_tensor_dequantize(qw_block, scale[i, j])
  return out

def blockwise_quantize(weight: torch.Tensor, block_size: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
  assert weight.ndim == block_size.numel() and weight.ndim == 2
  scale_shape = torch.Size((torch.tensor(list(weight.shape)) / block_size).ceil().long())
  scale = torch.empty(scale_shape, dtype=torch.float32)
  out = torch.empty_like(weight, dtype=dtype)
  for i in range(scale.shape[0]):
    for j in range(scale.shape[1]):
      block_size_i = block_size[0]
      block_size_j = block_size[1]
      w_block = weight[i*block_size_i:(i+1)*block_size_i, j*block_size_j:(j+1)*block_size_j]
      qw_block, scale_block = per_tensor_quantize(w_block, dtype)
      out[i*block_size_i:(i+1)*block_size_i, j*block_size_j:(j+1)*block_size_j] = qw_block
      scale[i, j] = scale_block
  return out, scale

def k_quantize(weight: torch.Tensor, method: Literal["q2_k"]) -> torch.Tensor:
  if method == "q2_k":
    return quantize_q2_k(weight)
  else:
    raise ValueError(f"Unsupported quantization method: {method}")

def per_expert_blockwise_quantize(expert_weights: torch.Tensor, block_size: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
  assert expert_weights.ndim == 3
  num_experts = expert_weights.shape[0]
  output_weights = []
  scales = []
  for e in range(num_experts):
    weight, scale = blockwise_quantize(expert_weights[e], block_size, dtype)
    output_weights.append(weight)
    scales.append(scale)
  return torch.stack(output_weights), torch.stack(scales)

def per_expert_k_quantize(expert_weights: torch.Tensor, method: Literal["q2_k"]) -> torch.Tensor:
  assert expert_weights.ndim == 3
  num_experts = expert_weights.shape[0]
  output_weights = []
  for e in range(num_experts):
    output_weights.append(k_quantize(expert_weights[e], method))
  return torch.stack(output_weights)

def load_weights(model_files: List[str], metadata: Metadata, tie_word_embeddings: bool, n_layers: int):
  """
  Generator that yields shards of weights loaded from the model files in huggingface format.
  Each shard contains a dictionary of tensors, with weights normalized and cast to the specified dtype
  (except layer norm weights which are converted to float32).
  """
  weights = {}
  for model_path in model_files:
    ext = os.path.splitext(model_path)[1]
    if ext == ".safetensors":
      with safetensors.safe_open(model_path, framework="pt") as f:
        for k in f.keys():
          assert(k not in weights)
          weights[k] = f.get_tensor(k)
  dtype = metadata.quant.dtype

  # convert weights
  progress = 0
  dequant_block_size = None
  if metadata.original_quantization_config is not None:
    dequant_block_size = torch.tensor(metadata.original_quantization_config["weight_block_size"])
  tensors = {}

  def load_and_dequantize(weight_name: str, scale_name: str) -> torch.Tensor:
    t = weights[weight_name]
    if scale_name in weights:
      scale = weights[scale_name]
      t = blockwise_dequantize(t, scale, dequant_block_size)
    return t
  
  def quantize(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if dtype not in [torch.float32, torch.float16]:
      if isinstance(metadata.quant, KQuant):
        t = k_quantize(t.to(torch.float32), metadata.quant.name)
      elif metadata.quant.block_size is None:
        return per_tensor_quantize(t, dtype)
      else:
        quant_block_size = torch.tensor(metadata.quant.block_size)
        return blockwise_quantize(t, quant_block_size, dtype)
    return t.to(dtype), None

  def conv(weight_name: str, scale_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    nonlocal progress
    progress += 1
    t = load_and_dequantize(weight_name, scale_name)
    print(f"\rConverting tensor {progress}: {t.shape}", end="", flush=True)
    return quantize(t)
  
  def conv_experts(weight_and_scale_names: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
    nonlocal progress
    progress += 1
    expert_weights = [weights[weight_name] for weight_name, _ in weight_and_scale_names]
    if weight_and_scale_names[0][1] in weights:
      for i in range(len(weight_and_scale_names)):
        scale = weights[weight_and_scale_names[i][1]]
        expert_weights[i] = blockwise_dequantize(expert_weights[i], scale, dequant_block_size)
    t = torch.stack(expert_weights)
    print(f"\rConverting tensor {progress}: {t.shape}", end="", flush=True)
    if dtype not in [torch.float32, torch.float16]:
      if isinstance(metadata.quant, KQuant):
        t = per_expert_k_quantize(t.to(torch.float32), metadata.quant.name)
      elif metadata.quant.block_size is None:
        return per_tensor_quantize(t, dtype)
      else:
        quant_block_size = torch.tensor(metadata.quant.block_size)
        return per_expert_blockwise_quantize(t, quant_block_size, dtype)
    return t.to(dtype), None
  
  def save_weight_and_scale(weight_name: str, scale_name: str, weight_and_scale: Tuple[torch.Tensor, torch.Tensor]):
    tensors[weight_name] = weight_and_scale[0]
    if weight_and_scale[1] is not None:
      tensors[scale_name] = weight_and_scale[1]

  save_weight_and_scale(
    "model.embed.weight", "model.embed.scale", 
    conv("model.embed_tokens.weight", "model.embed_tokens.weight_scale_inv")
  )

  for l in range(config["num_hidden_layers"]):
    if l % 8 == 0 and l > 0:
      yield tensors
      tensors = {}
    if n_layers is not None and l >= n_layers:
      break

    tensors[f"model.layers.{l}.attn.norm.weight"] = weights[f"model.layers.{l}.input_layernorm.weight"].float()
    tensors[f"model.layers.{l}.attn.kv_a_norm.weight"] = weights[f"model.layers.{l}.self_attn.kv_a_layernorm.weight"].float()

    if metadata.use_mla:
      assert metadata.q_lora_rank > 0
      head_dim = metadata.qk_nope_head_dim + metadata.qk_rope_head_dim
      save_weight_and_scale(
        f"model.layers.{l}.attn.wkv_a.weight", f"model.layers.{l}.attn.wkv_a.scale", 
        conv(f"model.layers.{l}.self_attn.kv_a_proj_with_mqa.weight", f"model.layers.{l}.self_attn.kv_a_proj_with_mqa.weight_scale_inv")
      )
      save_weight_and_scale(
        f"model.layers.{l}.attn.wq_a.weight", f"model.layers.{l}.attn.wq_a.scale", 
        conv(f"model.layers.{l}.self_attn.q_a_proj.weight", f"model.layers.{l}.self_attn.q_a_proj.weight_scale_inv")
      )
      tensors[f"model.layers.{l}.attn.q_a_norm.weight"] = weights[f"model.layers.{l}.self_attn.q_a_layernorm.weight"].float()
      # (n_heads, head_dim-qk_rope_head_dim+v_head_dim, kv_lora_rank)
      kv_b_proj = load_and_dequantize(
        f"model.layers.{l}.self_attn.kv_b_proj.weight", f"model.layers.{l}.self_attn.kv_b_proj.weight_scale_inv"
      ).reshape(
        metadata.n_heads, -1, metadata.kv_lora_rank
      )
      # (n_heads, head_dim, q_lora_rank)
      q_b_proj = load_and_dequantize(
        f"model.layers.{l}.self_attn.q_b_proj.weight", f"model.layers.{l}.self_attn.q_b_proj.weight_scale_inv"
      ).reshape(
        metadata.n_heads, -1, metadata.q_lora_rank
      )
      # (n_heads, head_dim-qk_rope_head_dim, kv_lora_rank)
      k_nope_b_proj = kv_b_proj[:, :head_dim-metadata.qk_rope_head_dim]
      # (n_heads * v_head_dim, kv_lora_rank)
      v_b_proj = kv_b_proj[:, head_dim-metadata.qk_rope_head_dim:].reshape(
        metadata.n_heads * metadata.v_head_dim, metadata.kv_lora_rank
      )
      # (n_heads, head_dim-qk_rope_head_dim, q_lora_rank)
      q_nope_b_proj = q_b_proj[:, :head_dim-metadata.qk_rope_head_dim]
      # (n_heads, qk_rope_head_dim, q_lora_rank)
      q_rope_b_proj = q_b_proj[:, head_dim-metadata.qk_rope_head_dim:]
      # (n_heads, kv_lora_rank, q_lora_rank)
      c_proj = torch.bmm(k_nope_b_proj.transpose(1, 2), q_nope_b_proj)
      
      # NOTE: k_rope gets split from kv_a, so there is no k_rope_b_proj
      save_weight_and_scale(
        f"model.layers.{l}.attn.wq_rope_b.weight", f"model.layers.{l}.attn.wq_rope_b.scale", 
        quantize(q_rope_b_proj.reshape(-1, q_rope_b_proj.shape[-1]))
      )
      save_weight_and_scale(
        f"model.layers.{l}.attn.wc.weight", f"model.layers.{l}.attn.wc.scale", 
        quantize(c_proj.reshape(-1, c_proj.shape[-1]))
      )
      
      save_weight_and_scale(
        f"model.layers.{l}.attn.wv_b.weight", f"model.layers.{l}.attn.wv_b.scale",
        quantize(v_b_proj)
      )
      save_weight_and_scale(
        f"model.layers.{l}.attn.wo.weight", f"model.layers.{l}.attn.wo.scale", 
        conv(f"model.layers.{l}.self_attn.o_proj.weight", f"model.layers.{l}.self_attn.o_proj.weight_scale_inv")
      )
    else:
      save_weight_and_scale(
        f"model.layers.{l}.attn.wkv_a.weight", f"model.layers.{l}.attn.wkv_a.scale", 
        conv(f"model.layers.{l}.self_attn.kv_a_proj_with_mqa.weight", f"model.layers.{l}.self_attn.kv_a_proj_with_mqa.weight_scale_inv")
      )
      save_weight_and_scale(
        f"model.layers.{l}.attn.wkv_b.weight", f"model.layers.{l}.attn.wkv_b.scale", 
        conv(f"model.layers.{l}.self_attn.kv_b_proj.weight", f"model.layers.{l}.self_attn.kv_b_proj.weight_scale_inv")
      )
      save_weight_and_scale(
        f"model.layers.{l}.attn.wo.weight", f"model.layers.{l}.attn.wo.scale", 
        conv(f"model.layers.{l}.self_attn.o_proj.weight", f"model.layers.{l}.self_attn.o_proj.weight_scale_inv")
      )
      if metadata.q_lora_rank > 0:
        save_weight_and_scale(
          f"model.layers.{l}.attn.wq_a.weight", f"model.layers.{l}.attn.wq_a.scale", 
          conv(f"model.layers.{l}.self_attn.q_a_proj.weight", f"model.layers.{l}.self_attn.q_a_proj.weight_scale_inv")
        )
        save_weight_and_scale(
          f"model.layers.{l}.attn.wq_b.weight", f"model.layers.{l}.attn.wq_b.scale", 
          conv(f"model.layers.{l}.self_attn.q_b_proj.weight", f"model.layers.{l}.self_attn.q_b_proj.weight_scale_inv")
        )
        tensors[f"model.layers.{l}.attn.q_a_norm.weight"] = weights[f"model.layers.{l}.self_attn.q_a_layernorm.weight"].float()
      else:
        save_weight_and_scale(
          f"model.layers.{l}.attn.wq.weight", f"model.layers.{l}.attn.wq.scale", 
          conv(f"model.layers.{l}.self_attn.q_proj.weight", f"model.layers.{l}.self_attn.q_proj.weight_scale_inv")
        )

    tensors[f"model.layers.{l}.mlp.norm.weight"] = weights[f"model.layers.{l}.post_attention_layernorm.weight"].float()

    if l < metadata.first_k_dense_replace:
      save_weight_and_scale(
        f"model.layers.{l}.mlp.w1.weight", f"model.layers.{l}.mlp.w1.scale", 
        conv(f"model.layers.{l}.mlp.gate_proj.weight", f"model.layers.{l}.mlp.gate_proj.weight_scale_inv")
      )
      save_weight_and_scale(
        f"model.layers.{l}.mlp.w2.weight", f"model.layers.{l}.mlp.w2.scale", 
        conv(f"model.layers.{l}.mlp.down_proj.weight", f"model.layers.{l}.mlp.down_proj.weight_scale_inv")
      )
      save_weight_and_scale(
        f"model.layers.{l}.mlp.w3.weight", f"model.layers.{l}.mlp.w3.scale", 
        conv(f"model.layers.{l}.mlp.up_proj.weight", f"model.layers.{l}.mlp.up_proj.weight_scale_inv")
      )
    else:
      tensors[f"model.layers.{l}.moegate.weight"] = weights[f"model.layers.{l}.mlp.gate.weight"].float()
      if metadata.arch == "DeepseekV3ForCausalLM":
        tensors[f"model.layers.{l}.moegate.bias"] = weights[f"model.layers.{l}.mlp.gate.e_score_correction_bias"].float()
      
      save_weight_and_scale(
        f"model.layers.{l}.mlp.w1.weight", f"model.layers.{l}.mlp.w1.scale", 
        conv_experts([
          (f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight", f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight_scale_inv") 
          for e in range(metadata.n_routed_experts)
        ])
      )
      save_weight_and_scale(
        f"model.layers.{l}.mlp.w2.weight", f"model.layers.{l}.mlp.w2.scale", 
        conv_experts([
          (f"model.layers.{l}.mlp.experts.{e}.down_proj.weight", f"model.layers.{l}.mlp.experts.{e}.down_proj.weight_scale_inv") 
          for e in range(metadata.n_routed_experts)
        ])
      )
      save_weight_and_scale(
        f"model.layers.{l}.mlp.w3.weight", f"model.layers.{l}.mlp.w3.scale", 
        conv_experts([
          (f"model.layers.{l}.mlp.experts.{e}.up_proj.weight", f"model.layers.{l}.mlp.experts.{e}.up_proj.weight_scale_inv") 
          for e in range(metadata.n_routed_experts)
        ])
      )
      save_weight_and_scale(
        f"model.layers.{l}.shared_mlp.w1.weight", f"model.layers.{l}.shared_mlp.w1.scale", 
        conv(f"model.layers.{l}.mlp.shared_experts.gate_proj.weight", f"model.layers.{l}.mlp.shared_experts.gate_proj.weight_scale_inv")
      )
      save_weight_and_scale(
        f"model.layers.{l}.shared_mlp.w2.weight", f"model.layers.{l}.shared_mlp.w2.scale", 
        conv(f"model.layers.{l}.mlp.shared_experts.down_proj.weight", f"model.layers.{l}.mlp.shared_experts.down_proj.weight_scale_inv")
      )
      save_weight_and_scale(
        f"model.layers.{l}.shared_mlp.w3.weight", f"model.layers.{l}.shared_mlp.w3.scale", 
        conv(f"model.layers.{l}.mlp.shared_experts.up_proj.weight", f"model.layers.{l}.mlp.shared_experts.up_proj.weight_scale_inv")
      )

  tensors["model.norm.weight"] = weights["model.norm.weight"].float()
  if tie_word_embeddings == False:
    save_weight_and_scale(
      "model.output.weight", "model.output.scale", 
      conv("lm_head.weight", "lm_head.weight_scale_inv")
    )
  else:
    # Model output classifier just uses the word embeddings matrix
    pass

  print()  # newline
  yield tensors

if __name__ == "__main__":
  argp = argparse.ArgumentParser()
  argp.add_argument("output_dir", type=str)
  argp.add_argument("input", type=str, nargs="?")
  argp.add_argument("--mla", action="store_true")
  argp.add_argument("--quant", type=str, default="fp16", choices=SUPPORTED_QUANTS)
  argp.add_argument("--n-layers", type=int, default=None, help="number of layers to convert (if None, convert all)")
  args = argp.parse_args()

  if os.path.exists(args.output_dir) and not os.path.isdir(args.output_dir):
    argp.error(f"output directory {args.output_dir} already exists and is not a directory")
  os.makedirs(args.output_dir, exist_ok=True)

  if args.input is not None:
    # Input is a directory with HuggingFace layout, e.g. files:
    #   config.json
    #   tokenizer.json
    #   *.safetensors
    args.config = os.path.join(args.input, "config.json")
    if not os.path.exists(args.config):
      argp.error(f"config.json not found in {args.input}")
    
    args.tokenizer = os.path.join(args.input, "tokenizer.json")
    if not os.path.exists(args.tokenizer):
      argp.error(f"tokenizer.json not found in {args.input}")
    
    files = os.listdir(args.input)
    args.models = [os.path.join(args.input, fname) for fname in files if os.path.splitext(fname)[1] == ".safetensors"]
    if len(args.models) == 0:
      argp.error(f"no .safetensors files found in {args.input}")
  else:
    argp.error("argument input is required")

  with open(args.config, "r") as f:
    config = json.load(f)
    metadata = Metadata(config, args.quant, args.n_layers, args.mla)

  tokens = load_tokens(args.tokenizer, metadata.vocab_size)
  
  # Process and save weight shards
  for shard_idx, shard in enumerate(load_weights(args.models, metadata, config.get("tie_word_embeddings", None), args.n_layers)):
    if shard_idx == 0:
      shard["tokenizer.tokens"] = torch.cat([torch.tensor([x for x in b] + [0], dtype=torch.uint8) for b in tokens])
      save_file(shard, os.path.join(args.output_dir, f"shard_{shard_idx:03d}.dseek"), metadata.to_dict())
    else:
      save_file(shard, os.path.join(args.output_dir, f"shard_{shard_idx:03d}.dseek"), {})
    print(f"\nSaved shard {shard_idx}", flush=True)