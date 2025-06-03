#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>

#include "fmt/format.h"

#include "codec.h"
#include "model.h"
#include "profile.h"
#include "sampler.h"
#include "time_utils.h"
#include "tokenizer.h"

void error_usage() {
  fprintf(stderr, "Usage:   main <checkpoint_dir> [options]\n");
  fprintf(stderr, "Example: main model_weights_dir/ -i \"Q: What is the meaning of life?\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -h Display this help message\n");
  fprintf(stderr, "  -L Locks model weights to RAM, disabling swap. Requires sudo.\n");
  fprintf(stderr, "  -m [completion,passkey,perplexity,interactive] which mode to run in (default - completion)\n");
  fprintf(stderr, "  -T <int> sliding window context length (0 - max)\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Perplexity mode options:\n");
  fprintf(stderr, "  Choose one:\n");
  fprintf(stderr, "    -i <string> input prompt\n");
  fprintf(stderr, "    -f <filepath> input file with prompt\n");
  fprintf(stderr, "    -w use wikitext as input\n");
  fprintf(stderr, "Completion mode options:\n");
  fprintf(stderr, "  -n <int>    number of steps to run for in completion mode, default 256. 0 = max_seq_len, -1 = infinite\n");
  fprintf(stderr, "  Choose one:\n");
  fprintf(stderr, "    -i <string> input prompt\n");
  fprintf(stderr, "    -t <float> temperature (default - 1.0)\n");
  fprintf(stderr, "    -p <float> p for top-p sampling (default - 0.95)\n");
  fprintf(stderr, "    -f <filepath> input file with prompt\n");
  fprintf(stderr, "Passkey mode options:\n");
  fprintf(stderr, "  -n <int>    number of junk lines to insert (default - 250)\n");
  fprintf(stderr, "  -l <int>    passkey position (-1 - random)\n");
  exit(1);
}

void help_usage_interactive() {
  fprintf(stderr, "Usage:   <mode> [options]\n");
  fprintf(stderr, "Example: c -i \"Q: What is the meaning of life?\"\n");
  fprintf(stderr, "Modes:\n");
  fprintf(stderr, "  h Display this help message\n");
  fprintf(stderr, "  c Completion - complete a single prompt \n");
  fprintf(stderr, "  p Perplexity - compute perplexity of a single prompt \n");
  fprintf(stderr, "  k Passkey - test passkey extraction \n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Perplexity mode options:\n");
  fprintf(stderr, "  Choose one:\n");
  fprintf(stderr, "    -i <string> input prompt\n");
  fprintf(stderr, "    -f <filepath> input file with prompt\n");
  fprintf(stderr, "    -w use wikitext as input\n");
  fprintf(stderr, "Completion mode options:\n");
  fprintf(stderr, "  -n <int>    number of steps to run for in completion mode, default 256. 0 = max_seq_len, -1 = infinite\n");
  fprintf(stderr, "  Choose one:\n");
  fprintf(stderr, "    -i <string> input prompt\n");
  fprintf(stderr, "    -t <float> temperature (default - 1.0)\n");
  fprintf(stderr, "    -p <float> p for top-p sampling (default - 0.95)\n");
  fprintf(stderr, "    -f <filepath> input file with prompt\n");
  fprintf(stderr, "Passkey mode options:\n");
  fprintf(stderr, "  -n <int>    number of junk lines to insert (default - 250)\n");
  fprintf(stderr, "  -l <int>    passkey position (-1 - random)\n");
}

struct Session {
  Session(const std::string& checkpoint_dir, bool lock_model_weights, int context, uint64_t sampler_seed):
    model_data(checkpoint_dir, lock_model_weights),
    model(model_data, context),
    state(model.config),
    sampler(model.config, sampler_seed),
    tokenizer(model_data) {}
  YALMData model_data;
  Model model;
  InferenceState state;
  Sampler sampler;
  Tokenizer tokenizer;
};

struct CompletionArgs {
  std::string prompt;
  int num_steps;
  float temperature = 1.0;
  float top_p = 0.95;
  // Returns true if args are valid, false otherwise
  bool parse_args(const std::vector<const char*>& args) {
    std::string prompt_path = "";
    for (size_t i = 0; i < args.size();) {
      // do some basic validation
      if (args[i][0] != '-') {
        return false;
      } // must start with dash
      if (strlen(args[i]) != 2) {
        return false;
      } // must be -x (one dash, one letter)

      // read in the args
      if (args[i][1] == 'h') {
        return false;
      } else if (args[i][1] == 'i') {
        if (i + 1 >= args.size()) {
          return false;
        }
        prompt = args[i + 1];
        i += 2;
      } else if (args[i][1] == 't') {
        if (i + 1 >= args.size()) {
          return false;
        }
        temperature = std::stof(args[i + 1]);
        i += 2;
      } else if (args[i][1] == 'p') {
        if (i + 1 >= args.size()) {
          return false;
        }
        top_p = std::stof(args[i + 1]);
        i += 2;
      } else if (args[i][1] == 'f') {
        if (i + 1 >= args.size()) {
          return false;
        }
        prompt_path = args[i + 1];
        i += 2;
      } else if (args[i][1] == 'n') {
        if (i + 1 >= args.size()) {
          return false;
        }
        num_steps = std::stoi(args[i + 1]);
        i += 2;
      } else {
        return false;
      }
    }
    int has_prompt = prompt.size() > 0 ? 1 : 0;
    int has_prompt_path = prompt_path.size() > 0 ? 1 : 0;
    if ((has_prompt + has_prompt_path) != 1) {
      return false;
    } else if (has_prompt_path) {
      std::ifstream file(prompt_path);
      if (!file.is_open()) {
        std::cerr << "Error: could not open file " << prompt_path << std::endl;
        return false;
      }

      std::stringstream buffer;
      buffer << file.rdbuf();
      prompt = buffer.str();
    }
    return true;
  }
};

struct PasskeyArgs {
  int n_junk;
  int passkey_pos;
  // Returns true if args are valid, false otherwise
  bool parse_args(const std::vector<const char*>& args) {
    for (size_t i = 2; i < args.size();) {
      // do some basic validation
      if (args[i][0] != '-') {
        return false;
      } // must start with dash
      if (strlen(args[i]) != 2) {
        return false;
      } // must be -x (one dash, one letter)

      // read in the args
      if (args[i][1] == 'h') {
        return false;
      } else if (args[i][1] == 'l') {
        if (i + 1 >= args.size()) {
          return false;
        }
        passkey_pos = std::stoi(args[i + 1]);
        i += 2;
      } else if (args[i][1] == 'n') {
        if (i + 1 >= args.size()) {
          return false;
        }
        n_junk = std::stoi(args[i + 1]);
        i += 2;
      } else {
        return false;
      }
    }
    if (passkey_pos != -1 && (passkey_pos >= n_junk || passkey_pos < 0)) {
      std::cerr << "Error: passkey position must be between 0 and " << n_junk - 1 << std::endl;
      return false;
    }
    return true;
  }
};

struct PerplexityArgs {
  std::string prompt;
  bool use_wikitext = false;
  // Returns true if args are valid, false otherwise
  bool parse_args(const std::vector<const char*>& args) {
    std::string prompt_path = "";
    for (size_t i = 0; i < args.size();) {
      // do some basic validation
      if (args[i][0] != '-') {
        return false;
      } // must start with dash
      if (strlen(args[i]) != 2) {
        return false;
      } // must be -x (one dash, one letter)

      // read in the args
      if (args[i][1] == 'h') {
        return false;
      } else if (args[i][1] == 'i') {
        if (i + 1 >= args.size()) {
          return false;
        }
        prompt = args[i + 1];
        i += 2;
      } else if (args[i][1] == 'f') {
        if (i + 1 >= args.size()) {
          return false;
        }
        prompt_path = args[i + 1];
        i += 2;
      } else if (args[i][1] == 'w') {
        use_wikitext = true;
        i += 1;
      } else {
        return false;
      }
    }
    int has_prompt = prompt.size() > 0 ? 1 : 0;
    int has_prompt_path = prompt_path.size() > 0 ? 1 : 0;
    int has_wikitext = use_wikitext ? 1 : 0;
    if ((has_prompt + has_prompt_path + has_wikitext) != 1) {
      std::cerr << "Error: must provide exactly one nonempty -i <input prompt> or -f <input filepath> or -w" << std::endl;
      return false;
    } else if (has_prompt_path) {
      std::ifstream file(prompt_path);
      if (!file.is_open()) {
        std::cerr << "Error: could not open file " << prompt_path << std::endl;
        return false;
      }

      std::stringstream buffer;
      buffer << file.rdbuf();
      prompt = buffer.str();
    }
    return true;
  }
};

std::vector<int> encode_prompt(const std::string& prompt, Tokenizer& tokenizer) {
  std::vector<int> encoding;
  {
    uint64_t encode_start_ms = get_timestamp_ms();
    encoding = tokenizer.encode(prompt, true);
    uint64_t encode_end_ms = get_timestamp_ms();

    std::cout << tokenizer.encoding_to_debug_string(encoding) << std::endl;
    uint64_t encoding_ms = encode_end_ms - encode_start_ms;
    std::cout << fmt::format(
      "Encoding stats: ({} tokens, throughput: {:.5}tok/s, latency: {:.5}s/tok, total: {:.5}s)\n",
      encoding.size(),
      encoding.size() / (encoding_ms / 1000.0),
      (encoding_ms / 1000.0) / encoding.size(),
      encoding_ms / 1000.0
    ) << std::endl;
  }
  return encoding;
}

void run_completion(
  Session& session,
  const std::string& prompt,
  int num_steps,
  float temperature,
  float top_p
) {
  Model& model = session.model;
  InferenceState& state = session.state;
  Sampler& sampler = session.sampler;
  Tokenizer& tokenizer = session.tokenizer;

  std::cout << "Model active bytes with full context window: " << model.active_bytes(model.config->max_seq_len) << std::endl;
  std::cout << "Model active bytes with no context: " << model.active_bytes(0) << std::endl;

  if (num_steps == 0) {
    // `-n 0` means use the full context length
    num_steps = model.config->max_seq_len;
  }

  {
    ProfileDisabledScope profile_disabled;
    std::cout << "Running warmup..." << std::endl;
    // Do one inference as warmup.
    // On CPU, this ensures all tensors are loaded into memory via mmap.
    model.forward(state, 0, 0);
    std::cout << "Warmup complete" << std::endl;
  }

  std::vector<int> encoding = encode_prompt(prompt, tokenizer);

  uint64_t start_ms = get_timestamp_ms();
  size_t read_bytes = 0;
  // Hydrate KV cache by forwarding model on all prompt tokens and discarding output.
  // This also generates output logits for the last token.
  for (size_t pos = 0; pos < encoding.size(); pos++) {
    ProfileScope scope(fmt::format("fwd_pos_{:03d}_hydrate", pos));
    int token_id = encoding[pos];
    InferenceMode inferMode = pos + 1 == encoding.size() ? 
      InferenceMode::OUTPUT_LOGITS : InferenceMode::HYDRATE_KV_CACHE;
    model.forward(state, token_id, pos, inferMode);
    read_bytes += model.active_bytes(pos);
  }
  uint64_t end_hydrate_ms = get_timestamp_ms();
  // For N steps:
  // - Sample + decode output logits
  // - Forward the model
  for (int i = 0; i < num_steps || num_steps == -1; i++) {
    int token_id = sampler.sample(state, temperature, top_p);
    std::string token_str = tokenizer.decode_one(encoding.back(), token_id);
    std::cout << token_str << std::flush;
    encoding.push_back(token_id);
    if (token_id == tokenizer.eos_id || token_id == tokenizer.eot_id) {
      break;
    }
    ProfileScope scope(fmt::format("fwd_pos_{:03d}_decode", encoding.size() - 1));
    model.forward(state, token_id, encoding.size() - 1);
    read_bytes += model.active_bytes(encoding.size() - 1);
  }
  std::cout << "\n" << std::endl;
  uint64_t end_ms = get_timestamp_ms();
  double elapsed_s = (end_ms - start_ms) / 1000.0;
  std::cout << fmt::format(
    "Generation stats:\n"
    "  {} tokens\n"
    "  throughput: {:.5}tok/s\n"
    "  latency: {:.5}s/tok\n"
    "  hydrate: {:.5}s\n"
    "  bandwidth: {:.5}GB/s\n"
    "  total: {:.5}s\n",
    encoding.size(),
    encoding.size() / elapsed_s,
    elapsed_s / encoding.size(),
    (end_hydrate_ms - start_ms) / 1000.0,
    ((double)read_bytes / 1e9) / elapsed_s,
    elapsed_s
  ) << std::endl;

#if PROFILE_ENABLED
  std::cout << "Profile total times (sec): " << std::endl;
  for (const auto& [key, value] : profile_times()) {
    std::cout << key << ": " << value << std::endl;
  }
#endif
}

std::vector<int> V2_ENCODED_WIKITEXT = {
  #include "wikitest.cat.1chunk.v2-encoded.txt"
};

std::vector<int> V3_ENCODED_WIKITEXT = {
  #include "wikitest.cat.1chunk.v3-encoded.txt"
};

void run_perplexity(
  Session& session,
  const std::vector<int>& encoding
) {
  Model& model = session.model;
  InferenceState& state = session.state;
  Sampler& sampler = session.sampler;

  std::cout << "Model active bytes with full context window: " << model.active_bytes(model.config->max_seq_len) << std::endl;

  {
    ProfileDisabledScope profile_disabled;
    std::cout << "Running warmup..." << std::endl;
    // Do one inference as warmup.
    // On CPU, this ensures all tensors are loaded into memory via mmap.
    model.forward(state, 0, 0);
    std::cout << "Warmup complete" << std::endl;
  }

  double sum_logprob = 0.0;
  double ss_logprob = 0.0;
  // Generates output logits for all tokens in the prompt and sum log probs to
  // compute perplexity.
  uint64_t start_ms = get_timestamp_ms();
  size_t read_bytes = 0;
  size_t N = encoding.size() - 1;
  for (size_t pos = 0; pos + 1 < encoding.size(); pos++) {
    std::cout << "\r Computing perplexity..." << pos + 1 << "/" << N << std::flush;
    
    int token_id = encoding[pos];
    model.forward(state, token_id, pos);
    read_bytes += model.active_bytes(pos);

    double logprob = std::log(sampler.sample_prob(encoding[pos + 1], state));
    sum_logprob += logprob;
    ss_logprob += logprob * logprob;
  }
  std::cout << std::endl;
  uint64_t end_ms = get_timestamp_ms();
  double elapsed_s = (end_ms - start_ms)/1000.0;
  double perplexity = std::exp(-sum_logprob / N);
  double perplexity_error = perplexity * std::sqrt(
    (ss_logprob - sum_logprob * sum_logprob / N) / N / N
  );
  std::cout << fmt::format(
    "Stats:\n"
    "  {} tokens\n"
    "  perplexity: {:.5} ± {:.5}\n"
    "  throughput: {:.5}tok/s\n"
    "  latency: {:.5}s/tok\n"
    "  bandwidth: {:.5}GB/s\n"
    "  total: {:.5}s\n",
    N,
    perplexity,
    perplexity_error,
    N / elapsed_s,
    elapsed_s / N,
    ((double)read_bytes / 1e9) / elapsed_s,
    elapsed_s
  ) << std::endl;
}

void run_passkey(
  Session& session,
  const int n_junk,
  const int passkey_pos
) {
  Model& model = session.model;
  InferenceState& state = session.state;
  Sampler& sampler = session.sampler;
  Tokenizer& tokenizer = session.tokenizer;

  std::cout << "Model active bytes with full context window: " << model.active_bytes(model.config->max_seq_len) << std::endl;

  const std::string PROMPT_PREFIX = 
    "There is an important info hidden inside a lot of irrelevant text. "
    "Find it and memorize them. I will quiz you about the important information there.";
  const std::string PROMPT_SUFFIX = " What is the pass key? The pass key is";

  const int passkey = std::rand() % 50000 + 1;
  const int pos = passkey_pos == -1 ? std::rand() % n_junk : passkey_pos;

  std::string prompt = PROMPT_PREFIX;
  for (int i = 0; i < n_junk; i++) {
    if (i % n_junk == pos) {
      prompt += " The pass key is " + std::to_string(passkey) + ". Remember it. " + std::to_string(passkey) + " is the pass key.";
    }
    prompt += " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.";
  }
  prompt += PROMPT_SUFFIX;

  std::vector<int> encoding;
  {
    uint64_t encode_start_ms = get_timestamp_ms();
    encoding = tokenizer.encode(prompt, true);
    uint64_t encode_end_ms = get_timestamp_ms();

    uint64_t encoding_ms = encode_end_ms - encode_start_ms;
    std::cout << fmt::format(
      "Encoding stats: ({} tokens, throughput: {:.5}tok/s, latency: {:.5}s/tok, total: {:.5}s)\n",
      encoding.size(),
      encoding.size() / (encoding_ms / 1000.0),
      (encoding_ms / 1000.0) / encoding.size(),
      encoding_ms / 1000.0
    ) << std::endl;
  }

  // Allow max 16 steps to generate passkey
  const size_t MAX_GENERATION_STEPS = 16;

  std::cout << fmt::format(
    "Passkey test:\n"
    "  prompt: {} tokens\n"
    "  passkey: {}\n"
    "  passkey token index: ~{}\n",
    encoding.size(),
    passkey,
    (int)(((float)pos) / n_junk * encoding.size())
  ) << std::endl;

  size_t N = encoding.size();
  for (size_t pos = 0; pos < N; pos++) {
    std::cout << "\r Running passkey test..." << pos + 1 << "/" << N << std::flush;
    int token_id = encoding[pos];
    InferenceMode inferMode = pos + 1 == N ? 
      InferenceMode::OUTPUT_LOGITS : InferenceMode::HYDRATE_KV_CACHE;
    model.forward(state, token_id, pos, inferMode);
  }
  std::cout << std::endl;
  std::cout << PROMPT_SUFFIX << std::flush;
  for (size_t pos = N; pos < N + MAX_GENERATION_STEPS; pos++) {
    int token_id = sampler.sample(state);
    std::string token_str = tokenizer.decode_one(encoding.back(), token_id);
    std::cout << token_str << std::flush;
    encoding.push_back(token_id);
    if (token_id == tokenizer.eos_id || token_id == tokenizer.eot_id) {
      break;
    }
    model.forward(state, token_id, pos);
  }
  std::cout << std::endl;
}

void run_interactive(Session& session) {
  std::string input = "";
  while (true) {
    std::cout << "> " << std::flush;
    std::getline(std::cin, input);
    if (input == "exit") {
      break;
    }
    // Split string by space
    std::vector<std::string> arg_strs;
    std::stringstream ss(input);
    std::string arg;
    while (ss >> arg) {
      if (arg_strs.size() > 0 && arg_strs[arg_strs.size() - 1].starts_with("\"") && !arg_strs[arg_strs.size() - 1].ends_with("\"")) {
        // Double quotes enclose strings that can contain spaces
        arg_strs[arg_strs.size() - 1] += " " + arg;
        if (arg.ends_with("\"")) {
          // Remove the double quotes
          arg_strs[arg_strs.size() - 1] = arg_strs[arg_strs.size() - 1].substr(1, arg_strs[arg_strs.size() - 1].size() - 2);
        }
      } else {
        arg_strs.push_back(arg);
      }
    }
    if (arg_strs.size() == 0) {
      help_usage_interactive();
      continue;
    }
    std::string mode = arg_strs[0];
    if (std::string("completion").starts_with(mode)) {
      mode = "completion";
    } else if (std::string("passkey").starts_with(mode) && mode != "p") {
      mode = "passkey";
    } else if (std::string("perplexity").starts_with(mode) && mode != "p") {
      mode = "perplexity";
    } else if (std::string("interactive").starts_with(mode)) {
      mode = "interactive";
    } else {
      help_usage_interactive();
      continue;
    }
    std::vector<const char*> args;
    for (size_t i = 1; i < arg_strs.size(); i++) {
      args.push_back(arg_strs[i].c_str());
    }
    if (mode == "completion") {
      CompletionArgs completion_args;
      if (!completion_args.parse_args(args)) {
        help_usage_interactive();
        continue;
      }
      run_completion(session, completion_args.prompt, completion_args.num_steps, completion_args.temperature, completion_args.top_p);
    } else if (mode == "passkey") {
      PasskeyArgs passkey_args;
      if (!passkey_args.parse_args(args)) {
        help_usage_interactive();
        continue;
      }
      run_passkey(session, passkey_args.n_junk, passkey_args.passkey_pos);
    } else if (mode == "perplexity") {
      PerplexityArgs perplexity_args;
      if (!perplexity_args.parse_args(args)) {
        help_usage_interactive();
        continue;
      }
      std::vector<int> encoding;
      if (perplexity_args.use_wikitext) {
        if (session.model_data.metadata.at("arch").get<std::string>() == "DeepseekV3ForCausalLM") {
          encoding = V3_ENCODED_WIKITEXT;
        } else {
          encoding = V2_ENCODED_WIKITEXT;
        }
      } else {
        encoding = encode_prompt(perplexity_args.prompt, session.tokenizer);
      }
      run_perplexity(session, encoding);
    }
  }
}

int main(int argc, char* argv[]) {
  std::vector<const char*> args(argv, argv + argc);
  std::vector<const char*> next_args;

  std::string checkpoint_dir = "";    // e.g. out/model.bin
  // Options
  std::string mode = "completion";     // completion, passkey, perplexity, or interactive
  int context = 0;
  bool lock_model_weights = false;

  if (args.size() >= 2) {
    checkpoint_dir = args[1];
  } else {
    error_usage();
  }

  // read in session args first, put everything else in next_args
  for (size_t i = 2; i < args.size();) {
    if (args[i][0] == '-' && strlen(args[i]) == 2) {
      if (args[i][1] == 'h') {
        error_usage();
      } else if (args[i][1] == 'L') {
        lock_model_weights = true;
        i += 1;
      } else if (args[i][1] == 'm') {
        if (i + 1 >= args.size()) {
          error_usage();
        }
        mode = args[i + 1];
        if (std::string("completion").starts_with(mode)) {
          mode = "completion";
        } else if (std::string("passkey").starts_with(mode) && mode != "p") {
          mode = "passkey";
        } else if (std::string("perplexity").starts_with(mode) && mode != "p") {
          mode = "perplexity";
        } else if (std::string("interactive").starts_with(mode)) {
          mode = "interactive";
        } else {
          error_usage();
        }
        i += 2;
      } else if (args[i][1] == 'T') {
        if (i + 1 >= args.size()) {
          error_usage();
        }
        context = std::stoi(args[i + 1]);
        i += 2;
      } else {
        next_args.push_back(args[i]);
        i += 1;
      }
    } else {
      next_args.push_back(args[i]);
      i += 1;
    }
  }

  if (mode == "completion") {
    CompletionArgs completion_args;
    if (!completion_args.parse_args(next_args)) {
      error_usage();
    }
    Session session(checkpoint_dir, lock_model_weights, context, get_timestamp_ms());
    run_completion(session, completion_args.prompt, completion_args.num_steps, completion_args.temperature, completion_args.top_p);
  } else if (mode == "passkey") {
    PasskeyArgs passkey_args;
    if (!passkey_args.parse_args(next_args)) {
      error_usage();
    }
    Session session(checkpoint_dir, lock_model_weights, context, get_timestamp_ms());
    run_passkey(session, passkey_args.n_junk, passkey_args.passkey_pos);
  } else if (mode == "perplexity") {
    PerplexityArgs perplexity_args;
    if (!perplexity_args.parse_args(next_args)) {
      error_usage();
    }
    Session session(checkpoint_dir, lock_model_weights, context, get_timestamp_ms());
    std::vector<int> encoding;
    if (perplexity_args.use_wikitext) {
      if (session.model_data.metadata.at("arch").get<std::string>() == "DeepseekV3ForCausalLM") {
        encoding = V3_ENCODED_WIKITEXT;
      } else {
        encoding = V2_ENCODED_WIKITEXT;
      }
    } else {
      encoding = encode_prompt(perplexity_args.prompt, session.tokenizer);
    }
    run_perplexity(session, encoding);
  } else if (mode == "interactive") {
    if (next_args.size() != 0) {
      error_usage();
    }
    Session session(checkpoint_dir, lock_model_weights, context, get_timestamp_ms());
    run_interactive(session);
  }

  return 0;
}