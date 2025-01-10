# Converts a model consisting of a huggingface config.json, tokenizer.json, and .safetensors weights into a .yalm file,
# which:
# - Normalizes the config to a common format in the header
# - Combines any safetensors shards
# - Reads the token vocabulary into a simpler format
# - Performs quantization to fp8 if specified

import argparse
import os
import json
import safetensors
from safetensors.torch import save_file
import torch

SUPPORTED_ARCHITECTURES = [
  # TODO: Llama (deepseek 1)?
  # TODO: DeepseekForCausalLM
  "DeepseekV2ForCausalLM",
  # TODO: DeepseekV3ForCausalLM
]
SUPPORTED_DTYPES = ["fp32", "fp16", "fp8"]

class Metadata:
  def __init__(self, config, dtype):
    arch = config["architectures"][0]
    if arch not in SUPPORTED_ARCHITECTURES:
      raise Exception(f"Architecture {arch} is not supported, must be one of {SUPPORTED_ARCHITECTURES}")
    self.arch = arch
    if dtype not in SUPPORTED_DTYPES:
      raise Exception(f"Data type {dtype} is not supported, must be one of {SUPPORTED_DTYPES}")
    self.dtype = dtype
    if arch in ["DeepseekV2ForCausalLM"]:
      self.dim = config["hidden_size"]
      self.hidden_dim = config["intermediate_size"]
      self.n_layers = config["num_hidden_layers"]
      self.n_heads = config["num_attention_heads"]
      self.n_kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
      self.vocab_size = config["vocab_size"]
      self.max_seq_len = config["max_position_embeddings"]
      self.bos_token_id = config["bos_token_id"]
      self.eos_token_id = config["eos_token_id"]
      self.rope_theta = config.get("rope_theta", 10000.0)
      self.norm_eps = config["rms_norm_eps"]
      self.norm_type = "rmsnorm"

      assert config.get("attention_bias", False) == False
      assert config.get("mlp_bias", False) == False

      assert config["hidden_act"] in ["gelu", "silu"]
      self.act_type = config["hidden_act"]
      self.first_k_dense_replace = config["first_k_dense_replace"]

      # multi-latent attention
      self.kv_lora_rank = config["kv_lora_rank"]
      self.q_lora_rank = config["q_lora_rank"]
      self.qk_nope_head_dim = config["qk_nope_head_dim"]
      self.qk_rope_head_dim = config["qk_rope_head_dim"]
      assert config.get("quantization_config", None) is None # TODO: support for Deepseek v3
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
      assert self.topk_method != "noaux_tc" # TODO: support for Deepseek v3
  
  def to_dict(self):
    result = {}
    result["arch"] = self.arch
    result["dtype"] = self.dtype
    if self.arch in ["DeepseekV2ForCausalLM"]:
      result["dim"] = str(self.dim)
      result["hidden_dim"] = str(self.hidden_dim)
      result["n_layers"] = str(self.n_layers)
      result["n_heads"] = str(self.n_heads)
      result["n_kv_heads"] = str(self.n_kv_heads)
      result["vocab_size"] = str(self.vocab_size)
      result["max_seq_len"] = str(self.max_seq_len)
      result["bos_token_id"] = str(self.bos_token_id)
      result["eos_token_id"] = str(self.eos_token_id)
      result["rope_theta"] = str(self.rope_theta)
      result["norm_eps"] = str(self.norm_eps)
      result["norm_type"] = str(self.norm_type)
      result["act_type"] = str(self.act_type)
      result["first_k_dense_replace"] = str(self.first_k_dense_replace)
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

def load_weights(model_files, dtype_str, metadata, tie_word_embeddings):
  """
  Load all weights from the model files in huggingface format into a dictionary of tensors,
  normalizing the attention weights, and casting all tensors (except for all layer norm weights,
  which are converted to float32) to the specified dtype.
  """
  weights = {}
  for model_path in model_files:
    ext = os.path.splitext(model_path)[1]
    if ext == ".safetensors":
      with safetensors.safe_open(model_path, framework="pt") as f:
        for k in f.keys():
          assert(k not in weights)
          weights[k] = f.get_tensor(k)
  dtype = {"fp32": torch.float32, "fp16": torch.float16, "fp8": torch.float8_e5m2}[dtype_str]

  # convert weights
  progress = 0
  def conv(t):
    nonlocal progress
    progress += 1
    print(f"\rConverting tensor {progress}: {t.shape}", end="", flush=True)
    return t.to(dtype)

  tensors = {}
  tensors["model.embed.weight"] = conv(weights["model.embed_tokens.weight"])

  for l in range(config["num_hidden_layers"]):
    tensors[f"model.layers.{l}.attn.norm.weight"] = weights[f"model.layers.{l}.input_layernorm.weight"].float()
    tensors[f"model.layers.{l}.attn.kv_a_norm.weight"] = weights[f"model.layers.{l}.self_attn.kv_a_layernorm.weight"].float()

    tensors[f"model.layers.{l}.attn.wkv_a.weight"] = conv(weights[f"model.layers.{l}.self_attn.kv_a_proj_with_mqa.weight"])
    tensors[f"model.layers.{l}.attn.wkv_b.weight"] = conv(weights[f"model.layers.{l}.self_attn.kv_b_proj.weight"])
    tensors[f"model.layers.{l}.attn.wo.weight"] = conv(weights[f"model.layers.{l}.self_attn.o_proj.weight"])
    if metadata.q_lora_rank > 0:
      tensors[f"model.layers.{l}.attn.wq_a.weight"] = conv(weights[f"model.layers.{l}.self_attn.q_a_proj.weight"])  
      tensors[f"model.layers.{l}.attn.wq_b.weight"] = conv(weights[f"model.layers.{l}.self_attn.q_b_proj.weight"])
      tensors[f"model.layers.{l}.attn.q_a_norm.weight"] = weights[f"model.layers.{l}.self_attn.q_a_layernorm.weight"].float()
    else:
      tensors[f"model.layers.{l}.attn.wq.weight"] = conv(weights[f"model.layers.{l}.self_attn.q_proj.weight"])

    tensors[f"model.layers.{l}.mlp.norm.weight"] = weights[f"model.layers.{l}.post_attention_layernorm.weight"].float()

    if l < metadata.first_k_dense_replace:
      tensors[f"model.layers.{l}.mlp.w1.weight"] = conv(weights[f"model.layers.{l}.mlp.gate_proj.weight"])
      tensors[f"model.layers.{l}.mlp.w2.weight"] = conv(weights[f"model.layers.{l}.mlp.down_proj.weight"])
      tensors[f"model.layers.{l}.mlp.w3.weight"] = conv(weights[f"model.layers.{l}.mlp.up_proj.weight"])
    else:
      tensors[f"model.layers.{l}.moegate.weight"] = conv(weights[f"model.layers.{l}.mlp.gate.weight"])
      tensors[f"model.layers.{l}.mlp.w1.weight"] = torch.stack([conv(weights[f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"]) for e in range(metadata.n_routed_experts)])
      tensors[f"model.layers.{l}.mlp.w2.weight"] = torch.stack([conv(weights[f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"]) for e in range(metadata.n_routed_experts)])
      tensors[f"model.layers.{l}.mlp.w3.weight"] = torch.stack([conv(weights[f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"]) for e in range(metadata.n_routed_experts)])
      tensors[f"model.layers.{l}.shared_mlp.w1.weight"] = conv(weights[f"model.layers.{l}.mlp.shared_experts.gate_proj.weight"])
      tensors[f"model.layers.{l}.shared_mlp.w2.weight"] = conv(weights[f"model.layers.{l}.mlp.shared_experts.down_proj.weight"])
      tensors[f"model.layers.{l}.shared_mlp.w3.weight"] = conv(weights[f"model.layers.{l}.mlp.shared_experts.up_proj.weight"])

  tensors["model.norm.weight"] = weights["model.norm.weight"].float()
  if tie_word_embeddings == False:
    tensors["model.output.weight"] = conv(weights["lm_head.weight"])
  else:
    # Model output classifier just uses the word embeddings matrix
    pass
  
  print() # newline
  return tensors

if __name__ == "__main__":
  argp = argparse.ArgumentParser()
  argp.add_argument("output", type=str)
  argp.add_argument("input", type=str, nargs="?")
  argp.add_argument("--dtype", type=str, default="fp16", choices=SUPPORTED_DTYPES)
  args = argp.parse_args()

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
    metadata = Metadata(config, args.dtype)

  tokens = load_tokens(args.tokenizer, metadata.vocab_size)
  tensors = load_weights(args.models, args.dtype, metadata, config.get("tie_word_embeddings", None))

  # add tokenizer tensors at the end (to maximize the chance of model tensor alignment)
  # note: we concatenate all bytes of all tokens into a single tensor
  tensors["tokenizer.tokens"] = torch.cat([torch.tensor([x for x in b] + [0], dtype=torch.uint8) for b in tokens])

  print(f"Saving {len(tensors)} tensors...")
  save_file(tensors, args.output, metadata.to_dict())