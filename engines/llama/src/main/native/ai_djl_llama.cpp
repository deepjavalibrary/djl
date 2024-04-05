#include <cstddef>
#include <iostream>
#include <mutex>
#include <string>

#include "ai_djl_llama_jni_LlamaLibrary.h"
#include "common.h"
#include "grammar-parser.h"
#include "llama.h"
#include "sampling.h"

// classes
static jclass c_lib_utils = 0;
static jclass c_model_params = 0;
static jclass c_input_params = 0;
static jclass c_token = 0;
static jclass c_standard_charsets = 0;
static jclass c_string = 0;
static jclass c_hash_map = 0;
static jclass c_map = 0;
static jclass c_set = 0;
static jclass c_entry = 0;
static jclass c_integer = 0;
static jclass c_float = 0;
static jclass c_logger = 0;
static jclass c_engine_exception = 0;

// constructors
static jmethodID cc_token = 0;
static jmethodID cc_hash_map = 0;
static jmethodID cc_integer = 0;
static jmethodID cc_float = 0;

// methods
static jmethodID m_get_bytes = 0;
static jmethodID m_entry_set = 0;
static jmethodID m_set_iterator = 0;
static jmethodID m_iterator_has_next = 0;
static jmethodID m_iterator_next = 0;
static jmethodID m_entry_key = 0;
static jmethodID m_entry_value = 0;
static jmethodID m_map_put = 0;
static jmethodID m_int_value = 0;
static jmethodID m_float_value = 0;
static jmethodID m_log_debug = 0;
static jmethodID m_log_info = 0;
static jmethodID m_log_warn = 0;
static jmethodID m_log_error = 0;

// fields
static jfieldID f_logger = 0;
// inference parameters
static jfieldID f_n_predict = 0;
static jfieldID f_n_keep = 0;
static jfieldID f_n_probs = 0;
static jfieldID f_logit_bias = 0;
static jfieldID f_top_k = 0;
static jfieldID f_top_p = 0;
static jfieldID f_tfs_z = 0;
static jfieldID f_typical_p = 0;
static jfieldID f_temperature = 0;
static jfieldID f_repeat_penalty = 0;
static jfieldID f_repeat_last_n = 0;
static jfieldID f_frequency_penalty = 0;
static jfieldID f_presence_penalty = 0;
static jfieldID f_penalize_nl = 0;
static jfieldID f_ignore_eos = 0;
static jfieldID f_mirostat = 0;
static jfieldID f_mirostat_tau = 0;
static jfieldID f_mirostat_eta = 0;
static jfieldID f_n_beams = 0;
static jfieldID f_grammar = 0;
static jfieldID f_antiprompt = 0;
static jfieldID f_infer_seed = 0;
// model parameters
static jfieldID f_n_threads = 0;
static jfieldID f_n_ctx = 0;
static jfieldID f_n_batch = 0;
static jfieldID f_n_gpu_layers = 0;
static jfieldID f_main_gpu = 0;
static jfieldID f_tensor_split = 0;
static jfieldID f_rope_freq_base = 0;
static jfieldID f_rope_freq_scale = 0;
static jfieldID f_mul_mat_q = 0; // unused since llamaCPP commit 3ab8b3a
static jfieldID f_f16_kv = 0;
static jfieldID f_logits_all = 0;
static jfieldID f_vocab_only = 0;
static jfieldID f_use_mmap = 0;
static jfieldID f_use_mlock = 0;
static jfieldID f_embedding = 0;
static jfieldID f_lora_adapter = 0;
static jfieldID f_lora_base = 0;
static jfieldID f_memory_f16 = 0;
static jfieldID f_mem_test = 0;
static jfieldID f_numa = 0;
static jfieldID f_verbose_prompt = 0;
// log level
static jfieldID f_utf_8 = 0;
// objects
static jobject o_utf_8 = 0;
static jobject o_logger = 0;

static JavaVM *g_vm = nullptr;

static void null_log_callback(enum ggml_log_level level, const char *text, void *user_data) {}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env = 0;

  if (JNI_OK != vm->GetEnv((void **) &env, JNI_VERSION_1_1)) {
    return JNI_ERR;
  }

  log_disable();
  llama_log_set(null_log_callback, nullptr);

  // find classes
  c_input_params = env->FindClass("ai/djl/llama/jni/InputParameters");
  c_model_params = env->FindClass("ai/djl/llama/jni/ModelParameters");
  c_lib_utils = env->FindClass("ai/djl/llama/jni/LibUtils");
  c_token = env->FindClass("ai/djl/llama/jni/Token");
  c_engine_exception = env->FindClass("ai/djl/engine/EngineException");
  c_logger = env->FindClass("org/slf4j/Logger");
  c_standard_charsets = env->FindClass("java/nio/charset/StandardCharsets");
  c_string = env->FindClass("java/lang/String");
  c_hash_map = env->FindClass("java/util/HashMap");
  c_map = env->FindClass("java/util/Map");
  c_set = env->FindClass("java/util/Set");
  c_entry = env->FindClass("java/util/Map$Entry");
  c_integer = env->FindClass("java/lang/Integer");
  c_float = env->FindClass("java/lang/Float");

  // create references
  c_input_params = (jclass) env->NewGlobalRef(c_input_params);
  c_model_params = (jclass) env->NewGlobalRef(c_model_params);
  c_lib_utils = (jclass) env->NewGlobalRef(c_lib_utils);
  c_token = (jclass) env->NewGlobalRef(c_token);
  c_engine_exception = (jclass) env->NewGlobalRef(c_engine_exception);
  c_logger = (jclass) env->NewGlobalRef(c_logger);
  c_string = (jclass) env->NewGlobalRef(c_string);
  c_hash_map = (jclass) env->NewGlobalRef(c_hash_map);
  c_map = (jclass) env->NewGlobalRef(c_map);
  c_set = (jclass) env->NewGlobalRef(c_set);
  c_entry = (jclass) env->NewGlobalRef(c_entry);
  c_integer = (jclass) env->NewGlobalRef(c_integer);
  c_float = (jclass) env->NewGlobalRef(c_float);

  // find constructors
  cc_token = env->GetMethodID(c_token, "<init>", "(I[BLjava/util/Map;JJZ)V");
  cc_hash_map = env->GetMethodID(c_hash_map, "<init>", "()V");
  cc_integer = env->GetMethodID(c_integer, "<init>", "(I)V");
  cc_float = env->GetMethodID(c_float, "<init>", "(F)V");

  // find methods
  m_get_bytes = env->GetMethodID(c_string, "getBytes", "(Ljava/lang/String;)[B");
  m_entry_set = env->GetMethodID(c_map, "entrySet", "()Ljava/util/Set;");
  m_entry_key = env->GetMethodID(c_entry, "getKey", "()Ljava/lang/Object;");
  m_entry_value = env->GetMethodID(c_entry, "getValue", "()Ljava/lang/Object;");
  m_map_put = env->GetMethodID(c_map, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
  m_int_value = env->GetMethodID(c_integer, "intValue", "()I");
  m_float_value = env->GetMethodID(c_float, "floatValue", "()F");
  m_log_debug = env->GetMethodID(c_logger, "debug", "(Ljava/lang/String;)V");
  m_log_info = env->GetMethodID(c_logger, "info", "(Ljava/lang/String;)V");
  m_log_warn = env->GetMethodID(c_logger, "warn", "(Ljava/lang/String;)V");
  m_log_error = env->GetMethodID(c_logger, "error", "(Ljava/lang/String;)V");

  // find fields
  f_logger = env->GetStaticFieldID(c_lib_utils, "logger", "Lorg/slf4j/Logger;");

  f_n_predict = env->GetFieldID(c_input_params, "nPredict", "I");
  f_n_keep = env->GetFieldID(c_input_params, "nKeep", "I");
  f_n_probs = env->GetFieldID(c_input_params, "nProbs", "I");
  f_logit_bias = env->GetFieldID(c_input_params, "logitBias", "Ljava/util/Map;");
  f_top_k = env->GetFieldID(c_input_params, "topK", "I");
  f_top_p = env->GetFieldID(c_input_params, "topP", "F");
  f_tfs_z = env->GetFieldID(c_input_params, "tfsZ", "F");
  f_typical_p = env->GetFieldID(c_input_params, "typicalP", "F");
  f_temperature = env->GetFieldID(c_input_params, "temperature", "F");
  f_repeat_penalty = env->GetFieldID(c_input_params, "repeatPenalty", "F");
  f_repeat_last_n = env->GetFieldID(c_input_params, "repeatLastN", "I");
  f_frequency_penalty = env->GetFieldID(c_input_params, "frequencyPenalty", "F");
  f_presence_penalty = env->GetFieldID(c_input_params, "presencePenalty", "F");
  f_penalize_nl = env->GetFieldID(c_input_params, "penalizeNl", "Z");
  f_ignore_eos = env->GetFieldID(c_input_params, "ignoreEos", "Z");
  f_mirostat = env->GetFieldID(c_input_params, "mirostat", "I");
  f_mirostat_tau = env->GetFieldID(c_input_params, "mirostatTau", "F");
  f_mirostat_eta = env->GetFieldID(c_input_params, "mirostatEta", "F");
  f_n_beams = env->GetFieldID(c_input_params, "nBeams", "I");
  f_grammar = env->GetFieldID(c_input_params, "grammar", "Ljava/lang/String;");
  f_antiprompt = env->GetFieldID(c_input_params, "antiPrompt", "[Ljava/lang/String;");
  f_infer_seed = env->GetFieldID(c_input_params, "seed", "I");

  f_n_threads = env->GetFieldID(c_model_params, "nThreads", "I");
  f_n_ctx = env->GetFieldID(c_model_params, "nCtx", "I");
  f_n_batch = env->GetFieldID(c_model_params, "nBatch", "I");
  f_n_gpu_layers = env->GetFieldID(c_model_params, "nGpuLayers", "I");
  f_main_gpu = env->GetFieldID(c_model_params, "mainGpu", "I");
  f_tensor_split = env->GetFieldID(c_model_params, "tensorSplit", "[F");
  f_rope_freq_base = env->GetFieldID(c_model_params, "ropeFreqBase", "F");
  f_rope_freq_scale = env->GetFieldID(c_model_params, "ropeFreqScale", "F");
  f_mul_mat_q = env->GetFieldID(c_model_params, "mulMatQ", "Z"); // unused since llamaCPP commit 3ab8b3a
  f_f16_kv = env->GetFieldID(c_model_params, "f16Kv", "Z");
  f_logits_all = env->GetFieldID(c_model_params, "logitsAll", "Z");
  f_vocab_only = env->GetFieldID(c_model_params, "vocabOnly", "Z");
  f_use_mmap = env->GetFieldID(c_model_params, "useMmap", "Z");
  f_use_mlock = env->GetFieldID(c_model_params, "useMlock", "Z");
  f_embedding = env->GetFieldID(c_model_params, "embedding", "Z");
  f_lora_adapter = env->GetFieldID(c_model_params, "loraAdapter", "Ljava/lang/String;");
  f_lora_base = env->GetFieldID(c_model_params, "loraBase", "Ljava/lang/String;");
  f_memory_f16 = env->GetFieldID(c_model_params, "memoryF16", "Z");
  f_mem_test = env->GetFieldID(c_model_params, "memTest", "Z");
  f_numa = env->GetFieldID(c_model_params, "numa", "I");
  f_verbose_prompt = env->GetFieldID(c_model_params, "verbosePrompt", "Z");

  f_utf_8 = env->GetStaticFieldID(c_standard_charsets, "UTF_8", "Ljava/nio/charset/Charset;");
  o_utf_8 = env->NewStringUTF("UTF-8");
  o_utf_8 = (jobject) env->NewGlobalRef(o_utf_8);
  o_logger = env->GetStaticObjectField(c_lib_utils, f_logger);
  o_logger = (jobject) env->NewGlobalRef(o_logger);

  if (env->ExceptionCheck()) {
    env->ExceptionDescribe();
    return JNI_ERR;
  }

  return JNI_VERSION_1_1;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
  JNIEnv *env = 0;

  if (JNI_OK != vm->GetEnv((void **) &env, JNI_VERSION_1_1)) {
    return;
  }

  env->DeleteGlobalRef(c_input_params);
  env->DeleteGlobalRef(c_model_params);
  env->DeleteGlobalRef(c_token);
  env->DeleteGlobalRef(c_string);
  env->DeleteGlobalRef(c_hash_map);
  env->DeleteGlobalRef(c_map);
  env->DeleteGlobalRef(c_set);
  env->DeleteGlobalRef(c_entry);
  env->DeleteGlobalRef(c_integer);
  env->DeleteGlobalRef(c_float);
  env->DeleteGlobalRef(c_logger);
  env->DeleteGlobalRef(c_engine_exception);

  env->DeleteGlobalRef(o_utf_8);
}

static void log(JNIEnv *env, enum ggml_log_level level, const char *text) {
  jstring java_text = env->NewStringUTF(text);

  switch (level) {
    case GGML_LOG_LEVEL_ERROR:
      env->CallVoidMethod(o_logger, m_log_error, java_text);
      break;
    case GGML_LOG_LEVEL_WARN:
      env->CallVoidMethod(o_logger, m_log_warn, java_text);
      break;
    case GGML_LOG_LEVEL_INFO:
      env->CallVoidMethod(o_logger, m_log_info, java_text);
      break;
    default:
      env->CallVoidMethod(o_logger, m_log_debug, java_text);
      break;
  }
  env->DeleteLocalRef(java_text);
}

static void log(JNIEnv *env, enum ggml_log_level level, std::string text) { log(env, level, text.c_str()); }

static std::string parse_jstring(JNIEnv *env, jstring java_string) {
  const jbyteArray string_bytes = (jbyteArray) env->CallObjectMethod(java_string, m_get_bytes, o_utf_8);

  size_t length = (size_t) env->GetArrayLength(string_bytes);
  jbyte *byte_elements = env->GetByteArrayElements(string_bytes, nullptr);

  std::string string = std::string((char *) byte_elements, length);

  env->ReleaseByteArrayElements(string_bytes, byte_elements, JNI_ABORT);
  env->DeleteLocalRef(string_bytes);

  return string;
}

static int parse_jinteger(JNIEnv *env, jobject java_integer) {
  if (!java_integer) return 0;
  return env->CallIntMethod(java_integer, m_int_value);
}

static float parse_jfloat(JNIEnv *env, jobject java_float) {
  if (!java_float) return 0;
  return env->CallFloatMethod(java_float, m_float_value);
}

static jbyteArray parse_jbytes(JNIEnv *env, std::string string) {
  jsize len = string.size();
  jbyteArray bytes = env->NewByteArray(len);
  env->SetByteArrayRegion(bytes, 0, len, reinterpret_cast<const jbyte *>(string.c_str()));
  return bytes;
}

// completion token output with probabilities
struct completion_token_output {
  struct token_prob {
    llama_token tok;
    float prob;
  };

  std::vector<token_prob> probs;
  llama_token tok;
};

static size_t common_part(const std::vector<llama_token> &a, const std::vector<llama_token> &b) {
  size_t i;
  for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {
  }
  return i;
}

enum stop_type {
  STOP_FULL,
  STOP_PARTIAL,
};

static bool ends_with(const std::string &str, const std::string &suffix) {
  return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop, const std::string &text) {
  if (!text.empty() && !stop.empty()) {
    const char text_last_char = text.back();
    for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
      if (stop[char_index] == text_last_char) {
        const std::string current_partial = stop.substr(0, char_index + 1);
        if (ends_with(text, current_partial)) {
          return text.size() - char_index - 1;
        }
      }
    }
  }
  return std::string::npos;
}

template <class Iter>
static std::string tokens_to_str(llama_context *ctx, Iter begin, Iter end) {
  std::string ret;
  for (; begin != end; ++begin) {
    ret += llama_token_to_piece(ctx, *begin);
  }
  return ret;
}

// format incomplete utf-8 multibyte character for output
static std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token) {
  std::string out = token == -1 ? "" : llama_token_to_piece(ctx, token);
  // if the size is 1 and first bit is 1, meaning it's a partial character
  //   (size > 1 meaning it's already a known token)
  if (out.size() == 1 && (out[0] & 0x80) == 0x80) {
    std::stringstream ss;
    ss << std::hex << (out[0] & 0xff);
    std::string res(ss.str());
    out = "byte: \\x" + res;
  }
  return out;
}

struct jllama_context {
  bool has_next_token = false;
  std::string generated_text;
  std::vector<completion_token_output> generated_token_probs;

  size_t num_prompt_tokens = 0;
  size_t num_tokens_predicted = 0;
  size_t n_past = 0;
  size_t n_remain = 0;

  std::string prompt;
  std::vector<llama_token> embd;
  std::vector<llama_token> last_n_tokens;

  llama_model *model = nullptr;
  llama_context *ctx = nullptr;
  gpt_params params;
  llama_sampling_context ctx_sampling;
  int n_ctx;

  grammar_parser::parse_state parsed_grammar;
  llama_grammar *grammar = nullptr;

  bool truncated = false;
  bool stopped_eos = false;
  bool stopped_word = false;
  bool stopped_limit = false;
  std::string stopping_word;
  int32_t multibyte_pending = 0;

  std::mutex mutex;

  std::unique_lock<std::mutex> lock() { return std::unique_lock<std::mutex>(mutex); }

  ~jllama_context() {
    if (ctx) {
      llama_free(ctx);
      ctx = nullptr;
    }
    if (model) {
      llama_free_model(model);
      model = nullptr;
    }
    if (grammar) {
      llama_grammar_free(grammar);
      grammar = nullptr;
    }
  }

  void rewind() {
    params.antiprompt.clear();
    params.sparams.grammar.clear();
    num_prompt_tokens = 0;
    num_tokens_predicted = 0;
    generated_text = "";
    generated_text.reserve(n_ctx);
    generated_token_probs.clear();
    truncated = false;
    stopped_eos = false;
    stopped_word = false;
    stopped_limit = false;
    stopping_word = "";
    multibyte_pending = 0;
    n_remain = 0;
    n_past = 0;

    if (grammar != nullptr) {
      llama_grammar_free(grammar);
      grammar = nullptr;
      ctx_sampling = *llama_sampling_init(params.sparams);
    }
  }

  bool loadModel(const gpt_params &params_) {
    params = params_;
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == nullptr) {
      return false;
    }
    n_ctx = llama_n_ctx(ctx);
    last_n_tokens.resize(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    return true;
  }

  std::vector<llama_token> tokenize(std::string prompt, bool add_bos) const {
    return ::llama_tokenize(ctx, prompt, add_bos);
  }

  bool loadGrammar(JNIEnv *env) {
    if (!params.sparams.grammar.empty()) {
      parsed_grammar = grammar_parser::parse(params.sparams.grammar.c_str());
      // will be empty (default) if there are parse errors
      if (parsed_grammar.rules.empty()) {
        log(env, GGML_LOG_LEVEL_ERROR, "grammar parse error");
        return false;
      }
      grammar_parser::print_grammar(stderr, parsed_grammar);

      {
        auto it = params.sparams.logit_bias.find(llama_token_eos(model));
        if (it != params.sparams.logit_bias.end() && it->second == -INFINITY) {
          log(env, GGML_LOG_LEVEL_WARN, "EOS token is disabled, which will cause most grammars to fail");
        }
      }

      std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
      grammar = llama_grammar_init(grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
    }
    ctx_sampling = *llama_sampling_init(params.sparams);
    return true;
  }

  void loadInfill(JNIEnv *env) {
    bool suff_rm_leading_spc = true;
    if (params.input_suffix.find_first_of(" ") == 0 && params.input_suffix.size() > 1) {
      params.input_suffix.erase(0, 1);
      suff_rm_leading_spc = false;
    }

    auto prefix_tokens = tokenize(params.input_prefix, false);
    auto suffix_tokens = tokenize(params.input_suffix, false);
    const int space_token = 29871;
    if (suff_rm_leading_spc && suffix_tokens[0] == space_token) {
      suffix_tokens.erase(suffix_tokens.begin());
    }
    prefix_tokens.insert(prefix_tokens.begin(), llama_token_prefix(model));
    prefix_tokens.insert(prefix_tokens.begin(), llama_token_bos(model));  // always add BOS
    prefix_tokens.insert(prefix_tokens.end(), llama_token_suffix(model));
    prefix_tokens.insert(prefix_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
    prefix_tokens.push_back(llama_token_middle(model));
    auto prompt_tokens = prefix_tokens;

    num_prompt_tokens = prompt_tokens.size();

    if (params.n_keep < 0) {
      params.n_keep = (int) num_prompt_tokens;
    }
    params.n_keep = std::min(params.n_ctx - 4, params.n_keep);

    // if input prompt is too big, truncate like normal
    if (num_prompt_tokens >= (size_t) params.n_ctx) {
      // todo we probably want to cut from both sides
      const int n_left = (params.n_ctx - params.n_keep) / 2;
      std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);
      const int erased_blocks = (num_prompt_tokens - params.n_keep - n_left - 1) / n_left;
      new_tokens.insert(
          new_tokens.end(), prompt_tokens.begin() + params.n_keep + erased_blocks * n_left, prompt_tokens.end());
      std::copy(prompt_tokens.end() - params.n_ctx, prompt_tokens.end(), last_n_tokens.begin());

      log(env, GGML_LOG_LEVEL_INFO, "input truncated n_left=" + std::to_string(n_left));

      truncated = true;
      prompt_tokens = new_tokens;
    } else {
      const size_t ps = num_prompt_tokens;
      std::fill(last_n_tokens.begin(), last_n_tokens.end() - ps, 0);
      std::copy(prompt_tokens.begin(), prompt_tokens.end(), last_n_tokens.end() - ps);
    }

    // compare the evaluated prompt with the new prompt
    n_past = common_part(embd, prompt_tokens);
    embd = prompt_tokens;

    if (n_past == num_prompt_tokens) {
      // we have to evaluate at least 1 token to generate logits.
      n_past--;
    }

    // since #3228 we now have to manually manage the KV cache
    llama_kv_cache_seq_rm(ctx, 0, n_past, -1);

    has_next_token = true;
  }

  void loadPrompt(JNIEnv *env) {
    auto prompt_tokens = tokenize(prompt, true);  // always add BOS

    num_prompt_tokens = prompt_tokens.size();

    if (params.n_keep < 0) {
      params.n_keep = (int) num_prompt_tokens;
    }
    params.n_keep = std::min(n_ctx - 4, params.n_keep);

    // if input prompt is too big, truncate like normal
    if (num_prompt_tokens >= (size_t) n_ctx) {
      const int n_left = (n_ctx - params.n_keep) / 2;
      std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);
      const int erased_blocks = (num_prompt_tokens - params.n_keep - n_left - 1) / n_left;
      new_tokens.insert(
          new_tokens.end(), prompt_tokens.begin() + params.n_keep + erased_blocks * n_left, prompt_tokens.end());
      std::copy(prompt_tokens.end() - n_ctx, prompt_tokens.end(), last_n_tokens.begin());

      log(env, GGML_LOG_LEVEL_INFO, "input truncated n_left=" + std::to_string(n_left));

      truncated = true;
      prompt_tokens = new_tokens;
    } else {
      const size_t ps = num_prompt_tokens;
      std::fill(last_n_tokens.begin(), last_n_tokens.end() - ps, 0);
      std::copy(prompt_tokens.begin(), prompt_tokens.end(), last_n_tokens.end() - ps);
    }

    // compare the evaluated prompt with the new prompt
    n_past = common_part(embd, prompt_tokens);

    embd = prompt_tokens;
    if (n_past == num_prompt_tokens) {
      // we have to evaluate at least 1 token to generate logits.
      n_past--;
    }

    // since #3228 we now have to manually manage the KV cache
    llama_kv_cache_seq_rm(ctx, 0, n_past, -1);

    has_next_token = true;
  }

  void beginCompletion() {
    // number of tokens to keep when resetting context
    n_remain = params.n_predict;
    llama_set_rng_seed(ctx, params.seed);
  }

  completion_token_output nextToken(JNIEnv *env) {
    completion_token_output result;
    result.tok = -1;

    if (embd.size() >= (size_t) n_ctx) {
      // Shift context

      const int n_left = n_past - params.n_keep - 1;
      const int n_discard = n_left / 2;

      llama_kv_cache_seq_rm(ctx, 0, params.n_keep + 1, params.n_keep + n_discard + 1);
      llama_kv_cache_seq_add(ctx, 0, params.n_keep + 1 + n_discard, n_past, -n_discard);

      for (size_t i = params.n_keep + 1 + n_discard; i < embd.size(); i++) {
        embd[i - n_discard] = embd[i];
      }
      embd.resize(embd.size() - n_discard);

      n_past -= n_discard;

      truncated = true;
      log(env, GGML_LOG_LEVEL_INFO, "input truncated n_left=" + std::to_string(n_left));
    }

    bool tg = true;
    while (n_past < embd.size()) {
      int n_eval = (int) embd.size() - n_past;
      tg = n_eval == 1;
      if (n_eval > params.n_batch) {
        n_eval = params.n_batch;
      }

      if (llama_decode(ctx, llama_batch_get_one(&embd[n_past], n_eval, n_past, 0))) {
        log(env, GGML_LOG_LEVEL_ERROR, "failed to eval n_eval=" + std::to_string(n_eval));
        has_next_token = false;
        return result;
      }
      n_past += n_eval;
    }

    if (params.n_predict == 0) {
      has_next_token = false;
      result.tok = llama_token_eos(model);
      return result;
    }

    {
      // out of user input, sample next token
      result.tok = llama_sampling_sample(&ctx_sampling, ctx, NULL);

      llama_token_data_array candidates_p = {ctx_sampling.cur.data(), ctx_sampling.cur.size(), false};

      const int32_t n_probs = params.sparams.n_probs;
      if (params.sparams.temp <= 0 && n_probs > 0) {
        // For llama_sample_token_greedy we need to sort candidates
        llama_sample_softmax(ctx, &candidates_p);
      }

      for (size_t i = 0; i < std::min(candidates_p.size, (size_t) n_probs); ++i) {
        result.probs.push_back({candidates_p.data[i].id, candidates_p.data[i].p});
      }

      llama_sampling_accept(&ctx_sampling, ctx, result.tok, true);
      if (tg) {
        num_tokens_predicted++;
      }
    }

    // add it to the context
    embd.push_back(result.tok);
    // decrement remaining sampling budget
    --n_remain;

    if (!embd.empty() && embd.back() == llama_token_eos(model)) {
      // stopping_word = llama_token_to_piece(ctx, embd.back());
      has_next_token = false;
      stopped_eos = true;
      return result;
    }

    has_next_token = params.n_predict == -1 || n_remain != 0;
    return result;
  }

  size_t findStoppingStrings(const std::string &text, const size_t last_token_size, const stop_type type) {
    size_t stop_pos = std::string::npos;
    for (const std::string &word : params.antiprompt) {
      size_t pos;
      if (type == STOP_FULL) {
        const size_t tmp = word.size() + last_token_size;
        const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
        pos = text.find(word, from_pos);
      } else {
        pos = find_partial_stop_string(word, text);
      }
      if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
        if (type == STOP_FULL) {
          stopping_word = word;
          stopped_word = true;
          has_next_token = false;
        }
        stop_pos = pos;
      }
    }
    return stop_pos;
  }

  completion_token_output doCompletion(JNIEnv *env) {
    auto token_with_probs = nextToken(env);

    const std::string token_text = token_with_probs.tok == -1 ? "" : llama_token_to_piece(ctx, token_with_probs.tok);
    generated_text += token_text;

    if (params.sparams.n_probs > 0) {
      generated_token_probs.push_back(token_with_probs);
    }

    if (multibyte_pending > 0) {
      multibyte_pending -= token_text.size();
    } else if (token_text.size() == 1) {
      const char c = token_text[0];
      // 2-byte characters: 110xxxxx 10xxxxxx
      if ((c & 0xE0) == 0xC0) {
        multibyte_pending = 1;
        // 3-byte characters: 1110xxxx 10xxxxxx 10xxxxxx
      } else if ((c & 0xF0) == 0xE0) {
        multibyte_pending = 2;
        // 4-byte characters: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
      } else if ((c & 0xF8) == 0xF0) {
        multibyte_pending = 3;
      } else {
        multibyte_pending = 0;
      }
    }

    if (multibyte_pending > 0 && !has_next_token) {
      has_next_token = true;
      n_remain++;
    }

    if (!has_next_token && n_remain == 0) {
      stopped_limit = true;
    }

    return token_with_probs;
  }

  std::vector<float> getEmbedding(JNIEnv *env) {
    static const int n_embd = llama_n_embd(model);
    if (!params.embedding) {
      log(env, GGML_LOG_LEVEL_ERROR, "embedding disabled");
      return std::vector<float>(n_embd, 0.0f);
    }
    const float *data = llama_get_embeddings(ctx);
    std::vector<float> embedding(data, data + n_embd);
    return embedding;
  }
};

static gpt_params parse_model_params(JNIEnv *env, jobject jparams, jstring java_file_path) {
  gpt_params params;

  params.model = parse_jstring(env, java_file_path);
  params.n_threads = env->GetIntField(jparams, f_n_threads);
  params.n_ctx = env->GetIntField(jparams, f_n_ctx);
  params.n_batch = env->GetIntField(jparams, f_n_batch);
  params.n_gpu_layers = env->GetIntField(jparams, f_n_gpu_layers);
  params.main_gpu = env->GetIntField(jparams, f_main_gpu);
  params.rope_freq_base = env->GetFloatField(jparams, f_rope_freq_base);
  params.rope_freq_scale = env->GetFloatField(jparams, f_rope_freq_scale);
  params.embedding = env->GetBooleanField(jparams, f_embedding);
  params.escape = env->GetIntField(jparams, f_n_predict);
  params.use_mmap = env->GetBooleanField(jparams, f_use_mmap);
  params.use_mlock = env->GetBooleanField(jparams, f_use_mlock);
  params.numa = (enum ggml_numa_strategy) env->GetIntField(jparams, f_numa); 
  params.verbose_prompt = env->GetBooleanField(jparams, f_verbose_prompt);
  
  if (params.model_alias == "unknown") {
    params.model_alias = params.model;
  }

  return params;
}

static void setup_infer_params(JNIEnv *env, jllama_context *llama, jobject jparams) {
  auto &params = llama->params;

  params.seed = env->GetIntField(jparams, f_infer_seed);
  params.n_predict = env->GetIntField(jparams, f_n_predict);
  params.n_keep = env->GetIntField(jparams, f_n_keep);

  auto &sparams = params.sparams;

  sparams.top_k = env->GetIntField(jparams, f_top_k);
  sparams.top_p = env->GetFloatField(jparams, f_top_p);
  sparams.tfs_z = env->GetFloatField(jparams, f_tfs_z);
  sparams.typical_p = env->GetFloatField(jparams, f_typical_p);
  sparams.temp = env->GetFloatField(jparams, f_temperature);
  sparams.penalty_repeat = env->GetFloatField(jparams, f_repeat_penalty);
  sparams.n_prev = env->GetIntField(jparams, f_repeat_last_n);
  sparams.penalty_freq = env->GetFloatField(jparams, f_frequency_penalty);
  sparams.penalty_present = env->GetFloatField(jparams, f_presence_penalty);
  sparams.penalize_nl = env->GetBooleanField(jparams, f_penalize_nl);
  sparams.mirostat = env->GetIntField(jparams, f_mirostat);
  sparams.mirostat_tau = env->GetFloatField(jparams, f_mirostat_tau);
  sparams.mirostat_eta = env->GetFloatField(jparams, f_mirostat_eta);
  sparams.n_probs = env->GetIntField(jparams, f_n_probs);

  jstring j_grammar = (jstring) env->GetObjectField(jparams, f_grammar);
  if (j_grammar != nullptr) {
    sparams.grammar = parse_jstring(env, j_grammar);
    env->DeleteLocalRef(j_grammar);
    if (!llama->loadGrammar(env)) {
      env->ThrowNew(c_engine_exception, "could not load grammar");
    }
  }

  sparams.logit_bias.clear();
  jboolean ignore_eos = env->GetBooleanField(jparams, f_ignore_eos);
  if (ignore_eos) {
    sparams.logit_bias[llama_token_eos(llama->model)] = -INFINITY;
  }

  jobject logit_bias = env->GetObjectField(jparams, f_logit_bias);
  if (logit_bias != nullptr) {
    jobject entry_set = env->CallObjectMethod(logit_bias, m_entry_set);
    jobject iterator = env->CallObjectMethod(entry_set, m_set_iterator);
    while (env->CallBooleanMethod(iterator, m_iterator_has_next)) {
      jobject entry = env->CallObjectMethod(iterator, m_iterator_next);
      jobject key = env->CallObjectMethod(entry, m_entry_key);
      jobject value = env->CallObjectMethod(entry, m_entry_value);

      int tok = parse_jinteger(env, key);
      float bias = parse_jfloat(env, value);
      sparams.logit_bias[tok] = bias;

      env->DeleteLocalRef(entry);
      env->DeleteLocalRef(key);
      env->DeleteLocalRef(value);
    }
  }

  params.antiprompt.clear();
  jobjectArray antiprompt = (jobjectArray) env->GetObjectField(jparams, f_antiprompt);
  if (antiprompt != nullptr) {
    jsize array_length = env->GetArrayLength(antiprompt);
    for (jsize i = 0; i < array_length; i++) {
      jstring java_string = (jstring) env->GetObjectArrayElement(antiprompt, i);
      if (java_string != nullptr) {
        std::string string = parse_jstring(env, java_string);
        params.antiprompt.push_back(string);
        env->DeleteLocalRef(java_string);
      }
    }
  }

  llama->ctx_sampling = *llama_sampling_init(params.sparams);
}

static void setup_answering(JNIEnv *env, jllama_context *llama, jstring prompt, jobject params) {
  llama->prompt = parse_jstring(env, prompt);
  llama->params.input_prefix = "";
  llama->params.input_suffix = "";
  setup_infer_params(env, llama, params);
}

static void setup_infilling(JNIEnv *env, jllama_context *llama, jstring prefix, jstring suffix, jobject params) {
  llama->prompt = "";
  llama->params.input_prefix = parse_jstring(env, prefix);
  llama->params.input_suffix = parse_jstring(env, suffix);
  setup_infer_params(env, llama, params);
}

JNIEXPORT jlong JNICALL Java_ai_djl_llama_jni_LlamaLibrary_loadModel(
    JNIEnv *env, jclass clazz, jstring file_path, jobject jparams) {
  gpt_params params = parse_model_params(env, jparams, file_path);

  jllama_context *llama = new jllama_context;
  llama_backend_init();

  if (!llama->loadModel(params)) {
    env->ThrowNew(c_engine_exception, "could not load model from given file path");
    return 0;
  }

  return reinterpret_cast<jlong>(llama);
}

JNIEXPORT void JNICALL Java_ai_djl_llama_jni_LlamaLibrary_generate(
    JNIEnv *env, jclass clazz, jlong handle, jstring prompt, jobject params) {
  auto *llama = reinterpret_cast<jllama_context *>(handle);

  llama->rewind();
  llama_reset_timings(llama->ctx);
  setup_answering(env, llama, prompt, params);

  llama->loadPrompt(env);
  llama->beginCompletion();
}

JNIEXPORT void JNICALL Java_ai_djl_llama_jni_LlamaLibrary_infill(
    JNIEnv *env, jclass clazz, jlong handle, jstring prefix, jstring suffix, jobject params) {
  auto *llama = reinterpret_cast<jllama_context *>(handle);

  llama->rewind();

  llama_reset_timings(llama->ctx);

  setup_infilling(env, llama, prefix, suffix, params);

  llama->loadInfill(env);
  llama->beginCompletion();
}

JNIEXPORT jobject JNICALL Java_ai_djl_llama_jni_LlamaLibrary_getNext(
    JNIEnv *env, jclass clazz, jlong handle, jlong sent_count, jlong sent_token_probs_index) {
  auto *llama = reinterpret_cast<jllama_context *>(handle);

  completion_token_output token_with_probs;
  while (llama->has_next_token) {
    token_with_probs = llama->doCompletion(env);
    if (token_with_probs.tok >= 0 && llama->multibyte_pending <= 0) {
      break;
    }
  }
  const std::string token_text = llama_token_to_piece(llama->ctx, token_with_probs.tok);

  size_t pos = std::min((size_t) sent_count, llama->generated_text.size());

  const std::string str_test = llama->generated_text.substr(pos);
  bool is_stop_full = false;
  size_t stop_pos = llama->findStoppingStrings(str_test, token_text.size(), STOP_FULL);
  if (stop_pos != std::string::npos) {
    is_stop_full = true;
    llama->generated_text.erase(llama->generated_text.begin() + pos + stop_pos, llama->generated_text.end());
    pos = std::min((size_t) sent_count, llama->generated_text.size());
  } else {
    is_stop_full = false;
    stop_pos = llama->findStoppingStrings(str_test, token_text.size(), STOP_PARTIAL);
  }

  std::string to_send;
  if (stop_pos == std::string::npos ||
      // Send rest of the text if we are at the end of the generation
      (!llama->has_next_token && !is_stop_full && stop_pos > 0)) {
    to_send = llama->generated_text.substr(pos, std::string::npos);

    sent_count += to_send.size();
    std::vector<completion_token_output> probs_output = {};

    if (llama->params.sparams.n_probs > 0) {
      const std::vector<llama_token> to_send_toks = llama_tokenize(llama->ctx, to_send, false);
      size_t probs_pos = std::min((size_t) sent_token_probs_index, llama->generated_token_probs.size());
      size_t probs_stop_pos =
          std::min(sent_token_probs_index + to_send_toks.size(), llama->generated_token_probs.size());
      if (probs_pos < probs_stop_pos) {
        probs_output = std::vector<completion_token_output>(
            llama->generated_token_probs.begin() + probs_pos, llama->generated_token_probs.begin() + probs_stop_pos);
      }
      sent_token_probs_index = probs_stop_pos;
    }
  } else {
    to_send = "";
  }

  jobject o_probabilities = env->NewObject(c_hash_map, cc_hash_map);
  for (const auto &tp : token_with_probs.probs) {
    jobject jtoken = env->NewObject(c_integer, cc_integer, tp.tok);
    jobject jprob = env->NewObject(c_float, cc_float, tp.prob);
    env->CallObjectMethod(o_probabilities, m_map_put, jtoken, jprob);
  }

  jbyteArray jbytes = parse_jbytes(env, to_send);
  return env->NewObject(c_token, cc_token, token_with_probs.tok, jbytes, o_probabilities, sent_count,
      sent_token_probs_index, llama->has_next_token);
}

JNIEXPORT jfloatArray JNICALL Java_ai_djl_llama_jni_LlamaLibrary_embed(
    JNIEnv *env, jclass clazz, jlong handle, jstring java_prompt) {
  auto *llama = reinterpret_cast<jllama_context *>(handle);

  llama->rewind();
  llama_reset_timings(llama->ctx);
  llama->prompt = parse_jstring(env, java_prompt);
  llama->params.n_predict = 0;
  llama->loadPrompt(env);
  llama->beginCompletion();
  llama->doCompletion(env);

  static const int n_embd = llama_n_embd(llama->model);
  const float *data = llama_get_embeddings(llama->ctx);
  std::vector<float> embedding(data, data + n_embd);

  jfloatArray java_embedding = env->NewFloatArray(embedding.size());
  env->SetFloatArrayRegion(java_embedding, 0, embedding.size(), reinterpret_cast<const jfloat *>(embedding.data()));

  return java_embedding;
}

JNIEXPORT jintArray JNICALL Java_ai_djl_llama_jni_LlamaLibrary_encode(
    JNIEnv *env, jclass clazz, jlong handle, jstring jprompt) {
  auto *llama = reinterpret_cast<jllama_context *>(handle);

  std::string prompt = parse_jstring(env, jprompt);
  std::vector<llama_token> tokens = llama->tokenize(prompt, false);

  jintArray java_tokens = env->NewIntArray(tokens.size());
  env->SetIntArrayRegion(java_tokens, 0, tokens.size(), reinterpret_cast<const jint *>(tokens.data()));

  return java_tokens;
}

JNIEXPORT jbyteArray JNICALL Java_ai_djl_llama_jni_LlamaLibrary_decodeBytes(
    JNIEnv *env, jclass clazz, jlong handle, jintArray java_tokens) {
  auto *llama = reinterpret_cast<jllama_context *>(handle);

  jsize length = env->GetArrayLength(java_tokens);
  jint *elements = env->GetIntArrayElements(java_tokens, nullptr);
  std::vector<llama_token> tokens(elements, elements + length);
  std::string text = tokens_to_str(llama->ctx, tokens.cbegin(), tokens.cend());

  env->ReleaseIntArrayElements(java_tokens, elements, 0);

  return parse_jbytes(env, text);
}

JNIEXPORT void JNICALL Java_ai_djl_llama_jni_LlamaLibrary_delete(JNIEnv *env, jclass clazz, jlong handle) {
  auto *llama = reinterpret_cast<jllama_context *>(handle);
  delete llama;
}
