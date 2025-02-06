This is an CPU-only inference implementation for the DeepSeek family of large language models written in C++, based on [Yet Another Language Model](https://github.com/andrewkchan/yalm). 

## Why?

For fun and learning!

I was initially adding DeepSeek support to `yalm` but realized that the changes were large and complex enough that it might ruin the simplicity of that project. Maybe at some point I'll upstream the changes, but for now I've decided to fork them into a separate, smaller, leaner codebase. 

Folks who want DeepSeek support on low-end CPU-only devices may also find this useful, especially since this program doesn't require a Python runtime and is tiny compared to other inference engines (<2k LOC not including `fmt` and `json`, vs. >250k for llama.cpp and vllm).

## Model support

| Model      | F8E5M2 | F8E4M3 | FP16 | BF16 | FP32 |
| -----      | ------ | ------ | ---- | ---- | ---- |
| DeepSeek-V2-Lite | ✅ | WIP | ✅ | WIP | ✅ |
| DeepSeek-V2 | ✅ | WIP | ✅ | WIP | ✅ |
| DeepSeek-V2.5 | ✅ | WIP | ✅ | WIP | ✅ |
| DeepSeek-V3 | ✅ | WIP | - | - | - |
| DeepSeek-R1 | ✅ | WIP | - | - | - |

# Instructions

deepseek.cpp requires a computer with a C++20-compatible compiler. You'll also need a directory containing LLM safetensor weights and configuration files in huggingface format, which you'll need to convert by providing a directory into which `.dseek` files containing the converted weights will go. Follow the below to download DeepSeek-V2-Lite, build `deepseek.cpp`, and run it:

```
# install git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get -y install git-lfs
# download DeepSeek-V2-Lite
git clone https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
# clone this repository
git clone https://github.com/andrewkchan/deepseek.cpp.git

cd deepseek.cpp
pip install -r requirements.txt
python convert.py --dtype fp16 v2-lite-f16 ../DeepSeek-V2-Lite/
./build/main v2-lite-f16 -i "What is a large language model?" -m c -t 1.0
```

## Usage

See the CLI help documentation below for `./build/main`:

```
Usage:   main <checkpoint_dir> [options]
Example: main model_weights_dir/ -i "Q: What is the meaning of life?"
Options:
  -h Display this help message
  -m [completion,passkey,perplexity] which mode to run in (default - completion)
  -T <int> sliding window context length (0 - max)

Perplexity mode options:
  Choose one:
    -i <string> input prompt
    -f <filepath> input file with prompt
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