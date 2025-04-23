## Spark-TTS 推理使用文档

> 本文档介绍如何使用 Spark-TTS 进行语音合成。

---

### 目录

1. [快速开始](#快速开始)
    - [初始化引擎](#初始化引擎)
    - [单句合成](#单句合成)
    - [长文本合成](#长文本合成)
    - [多角色合成](#多角色合成)
    - [语音克隆](#语音克隆)
2. [高级特性](#高级特性)
    - [音色复用（Acoustic Tokens）](#音色复用acoustic-tokens)
    - [流式输出](#流式输出)

3. [API 参考](#api-参考)

---

## 快速开始

以下示例基于 `AsyncSparkEngine` 接口：

### 初始化引擎

```python
from fast_tts import AsyncSparkEngine

engine = AsyncSparkEngine(
    model_path="checkpoints/Spark-TTS-0.5B",
    llm_device="cuda",
    tokenizer_device="cuda",
    detokenizer_device="cuda",
    backend="vllm",  # 可替换为 torch、vllm、sglang、llama-cpp、mlx-lm
)
```

### 单句合成

```python
import asyncio

text = "手握日月摘星辰，世间无我这般人。"
# 异步调用示例
wav = asyncio.run(engine.speak_async(text=text, name="female"))
# 保存音频
engine.write_audio(wav, "output.wav")
```

### 长文本合成

```python
import asyncio

long_text = "..."  # >200 字长文本
wav = asyncio.run(
    engine.speak_async(
        text=long_text,
        name="female",
        length_threshold=50,
        window_size=50
    )
)
engine.write_audio(wav, "long_output.wav")
```

### 多角色合成

```python
# 添加自定义角色
import asyncio


async def run():
    await engine.add_speaker("哪吒", audio="data/roles/哪吒.wav")
    await engine.add_speaker("李靖", audio="data/roles/李靖.wav")
    # 合成多角色对话
    multi_text = "<role:哪吒>...<role:李靖>..."
    wav = await engine.multi_speak_async(multi_text)
    engine.write_audio(wav, "multi.wav")


asyncio.run(run())

```

### 语音克隆

```python
wav = asyncio.run(
    engine.clone_voice_async(
        text="要克隆的文本",
        reference_audio="path/to/ref.wav",
        pitch="high",
        speed="moderate"
    )
)
engine.write_audio(wav, "cloned.wav")
```

## 高级特性

### 音色复用（Acoustic Tokens）

1. 初次合成返回 `SparkAcousticTokens`：
   ```python
   wav, tokens = asyncio.run(
       engine.speak_async(..., return_acoustic_tokens=True)
   )
   tokens.save("acoustic_tokens.txt")
   ```
2. 复用已保存音色：
   ```python
   from fast_tts import SparkAcousticTokens
   tokens = SparkAcousticTokens.load("acoustic_tokens.txt")
   wav2 = asyncio.run(
       engine.speak_async(..., acoustic_tokens=tokens)
   )
   ```

### 流式输出

```python
chunks = []
async for chunk in engine.speak_stream_async(text, name="female"):
    chunks.append(chunk)
audio = np.concatenate(chunks)
engine.write_audio(audio, "stream.wav")
```

## API 参考

详见 [fast_tts/engine/spark_engine](../../../fast_tts/engine/spark_engine.py) 下各方法注释，主要类和方法：

| 类 / 方法                     | 描述                      |
|----------------------------|-------------------------|
| `AsyncSparkEngine`         | 封装`Spark-TTS`初始化        |
| `speak_async`              | 异步文本合成，返回音频数据(np.int16) |
| `speak_stream_async`       | 异步流式合成，按音频块迭代返回         |
| `clone_voice_async`        | 异步语音克隆，基于参考音频生成语音       |
| `clone_voice_stream_async` | 异步流式语音克隆，基于参考音频生成流式语音   |

### `AsyncSparkEngine` 初始化参数说明

| 参数                            | 类型      | 默认值     | 描述                                                      |
|-------------------------------|---------|---------|---------------------------------------------------------|
| `model_path`                  | `str`   | —       | 模型根目录路径，包含 LLM、tokenizer、detokenizer 权重                 |
| `max_length`                  | `int`   | `32768` | LLM 最大上下文长度                                             |
| `llm_device`                  | `str`   | `auto`  | LLM模块计算设备                                               |
| `tokenizer_device`            | 同上      | `auto`  | 音频 tokenizer 计算设备                                       |
| `detokenizer_device`          | 同上      | `auto`  | 音频 detokenizer 计算设备                                     |
| `backend`                     | `str`   | `torch` | LLM加速后端，支持`torch`、`vllm`、`sglang`、`llama cpp`、`mlx-lm`  |
| `wav2vec_attn_implementation` | `str`   | `eager` | `wav2vec`模型的注意力算子，可选 `sdpa`、`flash_attention_2`、`eager` | 
| `llm_attn_implementation`     | 同上      | `eager` | LLM 注意力算子实现                                             |
| `torch_dtype`                 | `str`   | `auto`  | LLM模块权重量化类型选择，支持 `float16`、`bfloat16`、`float32`         |
| `llm_gpu_memory_utilization`  | `float` | `0.6`   | 后端显存占用率上限（仅 vllm/sglang 有效）                             |
| `batch_size`                  | `int`   | `1`     | 音频 tokenizer / detokenizer 并发批处理大小                      |
| `llm_batch_size`              | `int`   | `256`   | LLM 解码并行批处理大小                                           |
| `wait_timeout`                | `float` | `0.01`  | tokenizer / detokenizer 异步等待超时                          |
| `seed`                        | `int`   | `0`     | 随机种子                                                    |

### 主要接口参数说明

#### `speak_async`

| 参数                       | 类型                           | 默认值        | 描述                                                       |
|--------------------------|------------------------------|------------|----------------------------------------------------------|
| `text`                   | `str`                        | —          | 待合成文本                                                    |
| `name`                   | `str`                        | `female`   | 发音人角色（预设 female/male 或自定义）                               |
| `pitch`                  | `str`                        | `moderate` | 语调标签设置，可选`very_low`、`low`、`moderate`、、`high`、`very_high` |
| `speed`                  | 同上                           | `moderate` | 语速标签，同上                                                  |
| `temperature`            | `float`                      | `0.9`      | 采样温度                                                     |
| `top_k`                  | `int`                        | `50`       | top-k 采样                                                 |
| `top_p`                  | `float`                      | `0.95`     | top-p 采样                                                 |
| `repetition_penalty`     | `float`                      | `1.0`      | 重复惩罚系数                                                   |
| `max_tokens`             | `int`                        | `4096`     | LLM 最大生成 token 数                                         |
| `length_threshold`       | `int`                        | `50`       | 文本切分阈值，超过则按 `window_size` 分段                             |
| `window_size`            | `int`                        | `50`       | 文本滑动窗口大小                                                 |
| `split_fn`               | `Callable[[str], list[str]]` | `None`     | 自定义分割函数                                                  |
| `acoustic_tokens`        | `SparkAcousticTokens`或`str`  | `None`     | 复用的音色 tokens，仅针对female/male两种内置的角色                       | 
| `return_acoustic_tokens` | `bool`                       | `False`    | 是否返回初次生成的音色 tokens ，仅针对female/male两种内置的角色                |

#### `speak_stream_async`

与 `speak_async` 参数一致，额外支持：

| 参数                              | 类型      | 默认值   | 描述            |
|---------------------------------|---------|-------|---------------|
| `audio_chunk_duration`          | `float` | `1.0` | 每段音频时长（秒）     |
| `max_audio_chunk_duration`      | `float` | `8.0` | 最大音频块时长       |
| `audio_chunk_size_scale_factor` | `float` | `2.0` | 音频块时长扩展系数     |
| `audio_chunk_overlap_duration`  | `float` | `0.1` | 块间交叉淡入淡出时长（秒） |

#### `clone_voice_async`

| 参数                | 类型           | 默认值    | 描述                                  |
|-------------------|--------------|--------|-------------------------------------|
| `text`            | `str`        | —      | 待克隆文本                               |
| `reference_audio` | `bytes\|str` | —      | 参考音频文件路径或bytes                      | 参考音频路径或字节数据                    |
| `reference_text`  | `str`        | `None` | 参考音频的文字转录，可不传入                      |
| 其余同 `speak_async` |              |        | 如 `temperature`, `pitch`, `speed` 等 |

#### `clone_voice_stream_async`

参数与 `clone_voice_async` 对应，并支持流式输出的分块参数，与 `speak_stream_async` 保持一致：

| 参数                              | 类型                           | 默认值    | 描述                                                 |
|---------------------------------|------------------------------|--------|----------------------------------------------------|
| `text`                          | `str`                        | —      | 待克隆文本                                              |
| `reference_audio`               | `bytes\|str`                 | —      | 参考音频路径或字节数据                                        |
| `reference_text`                | `str`                        | `None` | 参考音频的文字转录                                          |
| 其他同 `speak_stream_async`        | —                            | —      | 包括 `temperature`、`top_k`、`top_p`、`pitch`、`speed` 等 |
| `split_fn`                      | `Callable[[str], list[str]]` | `None` | 自定义文本分割函数                                          |
| `audio_chunk_duration`          | `float`                      | `1.0`  | 每段音频时长（秒）                                          |
| `max_audio_chunk_duration`      | `float`                      | `8.0`  | 最大音频块时长                                            |
| `audio_chunk_size_scale_factor` | `float`                      | `2.0`  | 音频块时长扩展系数                                          |
| `audio_chunk_overlap_duration`  | `float`                      | `0.1`  | 块间交叉淡入淡出时长（秒）                                      |

