This is an CPU-only inference implementation for the DeepSeek family of large language models written in C++, based on [Yet Another Language Model](https://github.com/andrewkchan/yalm). 

## Why?

For fun and learning!

I was initially adding DeepSeek support to `yalm` but realized that the changes were large and complex enough that it might ruin the simplicity of that project. Maybe at some point I'll upstream the changes, but for now I've decided to fork them into a separate, smaller, leaner codebase. 

Since this program only supports DeepSeek, it's tiny compared to other inference engines (<2k LOC not including `fmt` and `json`, vs. >250k for llama.cpp and vllm) and is extra hackable. I'm currently using it as a testbed to study single-batch DeepSeek decoding performance on CPU.

## Model and hardware support

Quantizations other than FP32 require AVX2 and F16C support.

| Model      | Q2_K | Q3_K | Q4_K | F8E5M2 | F8E4M3 | FP16 | BF16 | FP32 |
| -----      | ---- | ---- | ------ | ------ | ---- | ---- | ---- | ---- |
| DeepSeek-V2-Lite | ✅ | ✅ | WIP | ✅ | WIP | ✅ | WIP | ✅ |
| DeepSeek-V2 | ✅ | ✅ | WIP | ✅ | WIP | ✅ | WIP | ✅ |
| DeepSeek-V2.5 | ✅ | ✅ | WIP | ✅ | WIP | ✅ | WIP | ✅ |
| DeepSeek-V3 | ✅ | ✅ | WIP | ✅ | WIP | - | - | - |
| DeepSeek-R1 | ✅ | ✅ | WIP | ✅ | WIP | - | - | - |

deepseek.cpp is missing important optimizations for production use (see notes below), but gets pretty close to llama.cpp in single-batch decode speed. Benchmarking DeepSeek-V3-Base with Q2_K quantization on an AWS r6a.12xlarge instance (AMD EPYC 7R13, 2x24 cores, 384GB DDR4 RAM):
- llama.cpp ([DeepSeek-V3-Q2_K_XS](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q2_K_XS) 207GB, tg128, best of 16/24/32/48 threads): 4.57 tok/s
- deepseek.cpp (Q2_K 207GB, MHA, `-n 128 -L` completion with 16 threads): 4.02 tok/s

A big part of this is that deepseek.cpp uses the llama.cpp vec_dot kernels for Q2_K, so I can't claim to have matched its performance purely through my own ingenuity. But it is surprising given the inference code is much simpler, opting for OpenMP over a [global threadpool with spinlock kernel barriers](https://justine.lol/matmul/#threads). I'm hoping that in addition to serving as a testbed for myself, this gives a good base for others to hack on.

# Instructions

deepseek.cpp requires a computer with a C++20-compatible compiler. You'll also need a directory containing LLM safetensor weights and configuration files in huggingface format, which you'll need to convert by providing a directory into which `.dseek` files containing the converted weights will go. Follow the below to download DeepSeek-V2-Lite, build `deepseek.cpp`, and run it:

```
# install git LFS and build tools
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get -y install git-lfs python3-dev build-essential
# download DeepSeek-V2-Lite
git clone https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
# clone this repository
git clone https://github.com/andrewkchan/deepseek.cpp.git

cd deepseek.cpp
pip install .
python convert.py --quant fp16 v2-lite-f16 ../DeepSeek-V2-Lite/
./build/main v2-lite-f16 -i "What is a large language model?" -m c -t 1.0
```

## Usage

See the CLI help documentation below for `./build/main`:

```
Usage:   main <checkpoint_dir> [options]
Example: main model_weights_dir/ -i "Q: What is the meaning of life?"
Options:
  -h Display this help message
  -L Locks model weights to RAM, disabling swap. Requires sudo.
  -m [completion,passkey,perplexity,interactive] which mode to run in (default - completion)
  -T <int> sliding window context length (0 - max)

Perplexity mode options:
  Choose one:
    -i <string> input prompt
    -f <filepath> input file with prompt
    -w use wikitext as input
Completion mode options:
  -n <int>    number of steps to run for in completion mode, default 256. 0 = max_seq_len, -1 = infinite
  Choose one:
    -i <string> input prompt
    -t <float> temperature (default - 1.0)
    -f <filepath> input file with prompt
Passkey mode options:
  -n <int>    number of junk lines to insert (default - 250)
  -l <int>    passkey position (-1 - random)
```

You will likely need to tune the number of OpenMP threads to achieve good performance. For example: 
```
OMP_NUM_THREADS=32 ./build/main <...args>
```

The default OpenMP thread count can result in severely degraded throughput, likely due to thread contention. I have found a good heuristic to be half the number of cores.

## Notes

- `--quant=f8e5m2` specifies model weight quantization using 128x128 blocks. MoE gates and layer norms are left in full precision. This should provide better accuracy than per-tensor quantization or the naive truncating quantization done by `yalm` (which results in nonsensical output for the DeepSeek family of models).
- `--quant=q2_k` and `--quant=q3_k` specify model weight quantization using the 2-bit and 3-bit llama.cpp [K-quantization schemes](https://github.com/ggml-org/llama.cpp/pull/1684), which use a two-level hierarchy of blocks and super-blocks to store scales/biases for ranges of weights.
- The models have a tendency to repeat themselves and get into infinite loops at lower temperatures. In my testing, a temperature of ~1.0 avoids this failure mode but also keeps the models reasonably grounded.
- Some new, optional architectural features (e.g. the `noaux_tc` method of expert selection) of DeepSeek V3 have not yet been implemented, so the model accuracy may be lower than the reference model.
- You will need ~650GB of memory to run DeepSeek V3 in F8E5M2, or 206GB for 2-bit Q2_K. For best performance, you should ensure there is enough physical RAM available and run as `sudo` with `-L` to force weights to stay in RAM, but otherwise, most operating systems will also automatically supplement this with swap space (storing some memory on disk and some in RAM) at the cost of severely degraded token throughput. More aggressive quantization methods such as [1.58-bit](https://unsloth.ai/blog/deepseekr1-dynamic) are planned.
- Model quality is not stable because I've been using this repository as an experiment testbed. See (https://github.com/andrewkchan/deepseek.cpp/pull/14) for the latest perplexity measurements on DeepSeek-V2-Lite as well as instructions on how to run standard measurements yourself. Known issues impacting generation quality include the tokenizer (which is not a true BPE tokenizer) and the use of attention sinks rather than yarn (https://github.com/andrewkchan/deepseek.cpp/pull/15).
- Only decoding (e.g. incremental, iterative generation or reading of one token at a time) has been implemented. Prefills (reading a batch of prompt tokens in a single pass) have not been implemented, nor prefill-based optimizations for the decoding phase such as speculative decoding or multi-token prediction. Finally, the current multi-latent attention implementation is still slower than multi-latent attention in surprising scenarios (https://github.com/andrewkchan/deepseek.cpp/pull/8) and appears to be under-utilizing memory bandwidth. I have limited time to implement these optimizations as this is a side project for me, but PRs are welcome!