## 快速推理

```bash
flashtts infer \
  -i "你好，欢迎使用语音合成。" \
  -o output.wav \
  -m ./models/your_model \
  -b vllm \
  [其他可选参数]
```

---

## `infer` 子命令参数

| 参数                  | 类型    | 默认           | 是否必需 | 说明                                   |
|---------------------|-------|--------------|------|--------------------------------------|
| `-i, --input`       | `str` | —            | 是    | 待合成文本，或者指向 `.txt` 的文件路径              |
| `-o, --output`      | `str` | `output.wav` | 否    | 合成后音频保存路径                            |
| `--name`            | `str` | `None`       | 否    | 内置角色名（使用角色语音而无需参考音频）                 |
| `--reference_audio` | `str` | `None`       | 否    | 参考音频路径（`.wav`），用于声线克隆                |
| `--reference_text`  | `str` | `None`       | 否    | 参考音频的文本（SparkTTS 模型克隆时必须提供）          |
| `--latent_file`     | `str` | `None`       | 否    | 参考音频的潜在向量 `.npy`（MegaTTS3 模型克隆时必须提供） |

> 若同时提供了 `--reference_audio`，会触发语音克隆功能；
> - SparkTTS 模型可增加参数（也可不填） `--reference_text`；
> - MegaTTS3 模型必须传入 `--latent_file`，且忽略 `--reference_text`。

---

## 模型加载参数（`add_model_parser`）

| 参数                              | 类型      | 默认      | 必需 | 说明                                                                          |
|---------------------------------|---------|---------|----|-----------------------------------------------------------------------------|
| `-m, --model_path`              | `str`   | —       | 是  | TTS 模型目录或权重文件路径                                                             |
| `-b, --backend`                 | `str`   | —       | 是  | 推理后端，可选 `llama-cpp, vllm, sglang, mlx-lm, torch`                            |
| `--lang`                        | `str`   | `None`  | 否  | OrpheusTTS 语言类型，如 `mandarin, english, french` 等                             |
| `--snac_path`                   | `str`   | `None`  | 否  | OrpheusTTS 专用 SNAC 模块路径                                                     |
| `--llm_device`                  | `str`   | `auto`  | 否  | LLM 运算设备，如 `cpu` 或 `cuda`                                                   |
| `--tokenizer_device`            | `str`   | `auto`  | 否  | Audio tokenizer 运算设备                                                        |
| `--detokenizer_device`          | `str`   | `auto`  | 否  | Audio detokenizer 运算设备                                                      |
| `--wav2vec_attn_implementation` | `str`   | `eager` | 否  | wav2vec 注意力实现：`sdpa, flash_attention_2, eager`                              |
| `--llm_attn_implementation`     | `str`   | `eager` | 否  | LLM 注意力实现：同上                                                                |
| `--max_length`                  | `int`   | `32768` | 否  | 最大生成长度（Token 数）                                                             |
| `--llm_gpu_memory_utilization`  | `float` | `0.6`   | 否  | `vllm`/`sglang` 后端 GPU 显存利用比例                                               |
| `--torch_dtype`                 | `str`   | `auto`  | 否  | Torch 后端数据类型：`float16, bfloat16, float32, auto`                             |
| `--cache_implementation`        | `str`   | `None`  | 否  | 解码缓存类型：`static, offloaded_static, sliding_window, hybrid, mamba, quantized` |
| `--seed`                        | `int`   | `0`     | 否  | 随机种子                                                                        |
| `--batch_size`                  | `int`   | `1`     | 否  | 单次合成最大并发请求数                                                                 |
| `--llm_batch_size`              | `int`   | `256`   | 否  | 单次 LLM 最大 batch 大小                                                          |
| `--wait_timeout`                | `float` | `0.01`  | 否  | 动态 batching 等待超时时间（秒）                                                       |

---

## 生成控制参数（`add_generate_parser`）

| 参数                     | 类型      | 默认     | 必需 | 说明                                              |
|------------------------|---------|--------|----|-------------------------------------------------|
| `--pitch`              | `str`   | `None` | 否  | 音高调整：`very_low, low, moderate, high, very_high` |
| `--speed`              | `str`   | `None` | 否  | 语速调整：同上                                         |
| `--temperature`        | `float` | `0.9`  | 否  | 随机性控制，越大生成越多样                                   |
| `--top_k`              | `int`   | `50`   | 否  | Top-K 采样保留最高概率前 K 个 token                       |
| `--top_p`              | `float` | `0.95` | 否  | Top-P（核采样）累积概率阈值                                |
| `--repetition_penalty` | `float` | `1.0`  | 否  | 重复惩罚系数，越大越抑制重复                                  |
| `--max_tokens`         | `int`   | `4096` | 否  | 最多生成 token 数                                    |

---

## 示例

1. **基础合成**
   ```bash
   flashtts infer \
     -i "Quick start demo." \
     -m ./models/spark-tts \
     -b vllm \
     -o demo.wav
   ```

2. **SparkTTS 声线克隆**
   ```bash
   flashtts infer \
     -i "克隆示例文本。" \
     -m ./models/spark-tts \
     -b vllm \
     --reference_audio ref.wav \
     -o clone.wav
   ```

3. **MegaTTS3 声线克隆**
   ```bash
   flashtts infer \
     -i "克隆示例文本。" \
     -m ./models/mega-tts3 \
     -b vllm \
     --reference_audio ref.wav \
     --latent_file ref_latent.npy \
     -o clone_mega.wav
   ```

4. **自定义音高与语速**
   ```bash
   flashtts infer \
     -i "调整音高和语速示例。" \
     -m ./models/spark-tts \
     -b vllm \
     --pitch high \
     --speed low \
     -o tuned.wav
   ```

---

## 常见问题

- **缺少 `--model_path`**  
  请使用 `-m/--model_path` 指定模型路径。
- **后端依赖未安装**  
  如果使用 `vllm`、`sglang` 等后端失败，确认已安装对应库。
- **MegaTTS3 报缺少 `latent_file`**  
  克隆 MegaTTS3 时，必须同时提供 `--latent_file`。
- **显存不足**  
  降低`--llm_gpu_memory_utilization`，或将audio处理模块放至CPU，或使用`llama-cpp`后端。  
