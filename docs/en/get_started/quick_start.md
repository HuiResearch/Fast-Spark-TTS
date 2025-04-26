## Quick Inference

```bash
flashtts infer \
  -i "Hello, welcome to speech synthesis." \
  -o output.wav \
  -m ./models/your_model \
  -b vllm \
  [other optional parameters]
```

---

## `infer` Subcommand Arguments

| Argument            | Type  | Default      | Required | Description                                                                 |
|---------------------|-------|--------------|----------|-----------------------------------------------------------------------------|
| `-i, --input`       | `str` | —            | Yes      | Input text or path to a `.txt` file                                         |
| `-o, --output`      | `str` | `output.wav` | No       | Output path for the synthesized audio                                       |
| `--name`            | `str` | `None`       | No       | Built-in character name (use character voice without reference audio)       |
| `--reference_audio` | `str` | `None`       | No       | Path to reference audio (`.wav`) for voice cloning                          |
| `--reference_text`  | `str` | `None`       | No       | Text of the reference audio (required for SparkTTS cloning)                 |
| `--latent_file`     | `str` | `None`       | No       | Latent vector `.npy` of the reference audio (required for MegaTTS3 cloning) |

> If `--reference_audio` is provided, voice cloning will be triggered:
> - For **SparkTTS**, optionally provide `--reference_text`.
> - For **MegaTTS3**, `--latent_file` must be provided and `--reference_text` is ignored.

---

## Model Loading Arguments (`add_model_parser`)

| Argument                        | Type    | Default | Required | Description                                                                               |
|---------------------------------|---------|---------|----------|-------------------------------------------------------------------------------------------|
| `-m, --model_path`              | `str`   | —       | Yes      | Path to the TTS model directory or weight file                                            |
| `-b, --backend`                 | `str`   | —       | Yes      | Inference backend: `llama-cpp, vllm, sglang, mlx-lm, torch`                               |
| `--lang`                        | `str`   | `None`  | No       | Language type for OrpheusTTS, e.g., `mandarin, english, french`, etc.                     |
| `--snac_path`                   | `str`   | `None`  | No       | Path to SNAC module for OrpheusTTS                                                        |
| `--llm_device`                  | `str`   | `auto`  | No       | Device for LLM computation: `cpu` or `cuda`                                               |
| `--tokenizer_device`            | `str`   | `auto`  | No       | Device for audio tokenizer                                                                |
| `--detokenizer_device`          | `str`   | `auto`  | No       | Device for audio detokenizer                                                              |
| `--wav2vec_attn_implementation` | `str`   | `eager` | No       | wav2vec attention implementation: `sdpa, flash_attention_2, eager`                        |
| `--llm_attn_implementation`     | `str`   | `eager` | No       | LLM attention implementation: same as above                                               |
| `--max_length`                  | `int`   | `32768` | No       | Maximum generation length (in tokens)                                                     |
| `--llm_gpu_memory_utilization`  | `float` | `0.6`   | No       | GPU memory utilization ratio for `vllm`/`sglang` backends                                 |
| `--torch_dtype`                 | `str`   | `auto`  | No       | Data type for Torch backend: `float16, bfloat16, float32, auto`                           |
| `--cache_implementation`        | `str`   | `None`  | No       | Decoding cache type: `static, offloaded_static, sliding_window, hybrid, mamba, quantized` |
| `--seed`                        | `int`   | `0`     | No       | Random seed                                                                               |
| `--batch_size`                  | `int`   | `1`     | No       | Max concurrent synthesis requests per batch                                               |
| `--llm_batch_size`              | `int`   | `256`   | No       | Max LLM batch size per run                                                                |
| `--wait_timeout`                | `float` | `0.01`  | No       | Timeout for dynamic batching (in seconds)                                                 |

---

## Generation Control Arguments (`add_generate_parser`)

| Argument               | Type    | Default | Required | Description                                                    |
|------------------------|---------|---------|----------|----------------------------------------------------------------|
| `--pitch`              | `str`   | `None`  | No       | Pitch adjustment: `very_low, low, moderate, high, very_high`   |
| `--speed`              | `str`   | `None`  | No       | Speed adjustment: same options as pitch                        |
| `--temperature`        | `float` | `0.9`   | No       | Controls randomness — higher values yield more diverse outputs |
| `--top_k`              | `int`   | `50`    | No       | Top-K sampling: retain top K tokens with highest probability   |
| `--top_p`              | `float` | `0.95`  | No       | Top-P (nucleus) sampling threshold                             |
| `--repetition_penalty` | `float` | `1.0`   | No       | Penalty factor for repetition; higher values reduce repetition |
| `--max_tokens`         | `int`   | `4096`  | No       | Maximum number of tokens to generate                           |

---

## Examples

1. **Basic Synthesis**
   ```bash
   flashtts infer \
     -i "Quick start demo." \
     -m ./models/spark-tts \
     -b vllm \
     -o demo.wav
   ```

2. **SparkTTS Voice Cloning**
   ```bash
   flashtts infer \
     -i "Voice cloning sample." \
     -m ./models/spark-tts \
     -b vllm \
     --reference_audio ref.wav \
     -o clone.wav
   ```

3. **MegaTTS3 Voice Cloning**
   ```bash
   flashtts infer \
     -i "Voice cloning sample." \
     -m ./models/mega-tts3 \
     -b vllm \
     --reference_audio ref.wav \
     --latent_file ref_latent.npy \
     -o clone_mega.wav
   ```

4. **Custom Pitch and Speed**
   ```bash
   flashtts infer \
     -i "Example with adjusted pitch and speed." \
     -m ./models/spark-tts \
     -b vllm \
     --pitch high \
     --speed low \
     -o tuned.wav
   ```

---

## FAQ

- **Missing `--model_path`**  
  Please specify the model path using `-m/--model_path`.
- **Backend Dependency Not Installed**  
  If you encounter backend failures (e.g., `vllm`, `sglang`), make sure the corresponding libraries are installed.
- **MegaTTS3 Missing `latent_file`**  
  When cloning with MegaTTS3, you must provide `--latent_file`.
- **Out of GPU Memory**  
  Try lowering `--llm_gpu_memory_utilization`, moving audio modules to CPU, or switching to the `llama-cpp` backend.
