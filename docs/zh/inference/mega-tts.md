## MegaTTS3 推理使用文档

> 本文档介绍如何使用 MegaTTS3 进行语音克隆。

---

### 目录

1. [快速开始](#快速开始)
    - [语音克隆](#语音克隆)

2. [API 参考](#api-参考)

---

## 快速开始

以下示例基于 `AsyncMega3Engine` 接口：

### 初始化引擎

```python
from fast_tts import AsyncMega3Engine

engine = AsyncMega3Engine(
    model_path="checkpoints/MegaTTS3",
    llm_device="cuda",
    tokenizer_device="cuda",
    backend="vllm",  # 可替换为 torch、vllm、sglang、llama-cpp、mlx-lm
)
```

### 语音克隆

```python
wav = asyncio.run(
    engine.clone_voice_async(
        text="另一边的桌上,一位读书人嗤之以鼻道,'佛子三藏,神子燕小鱼是什么样的人物,李家的那个李子夜如何与他们相提并论？",
        # mega tts模型的参考音频是一个元组，需要传入wav文件和编码后的npy文件  
        reference_audio=("data/mega-roles/蔡徐坤/蔡徐坤1.wav", "data/mega-roles/蔡徐坤/蔡徐坤1.npy"),
        max_tokens=512
    )
)
engine.write_audio(wav, "cloned.wav")
```

## API 参考

详见 [fast_tts/engine/mega_engine](../../../fast_tts/engine/mega_engine.py) 下各方法注释，主要类和方法：

| 类 / 方法              | 描述                                    |
|---------------------|---------------------------------------|
| `AsyncMega3Engine`  | 封装`MegaTTS3`初始化                       |
| `speak_async`       | 异步文本合成，返回音频数据(np.int16)，需要添加音频角色后才可使用 |
| `clone_voice_async` | 异步语音克隆，基于参考音频生成语音                     |

### `AsyncSparkEngine` 初始化参数说明

| 参数                           | 类型      | 默认值     | 描述                                                     |
|------------------------------|---------|---------|--------------------------------------------------------|
| `model_path`                 | `str`   | —       | 模型根目录路径                                                |
| `max_length`                 | `int`   | `32768` | LLM 最大上下文长度                                            |
| `llm_device`                 | `str`   | `auto`  | LLM模块计算设备                                              |
| `tokenizer_device`           | 同上      | `auto`  | 音频 tokenizer 计算设备                                      |
| `backend`                    | `str`   | `torch` | LLM加速后端，支持`torch`、`vllm`、`sglang`、`llama cpp`、`mlx-lm` |
| `llm_attn_implementation`    | 同上      | `eager` | LLM 注意力算子实现                                            |
| `torch_dtype`                | `str`   | `auto`  | LLM模块权重量化类型选择，支持 `float16`、`bfloat16`、`float32`        |
| `llm_gpu_memory_utilization` | `float` | `0.6`   | 后端显存占用率上限（仅 vllm/sglang 有效）                            |
| `batch_size`                 | `int`   | `1`     | 音频 tokenizer / detokenizer 并发批处理大小                     |
| `llm_batch_size`             | `int`   | `256`   | LLM 解码并行批处理大小                                          |
| `wait_timeout`               | `float` | `0.01`  | tokenizer / detokenizer 异步等待超时                         |
| `seed`                       | `int`   | `0`     | 随机种子                                                   |

### 主要接口参数说明

#### `speak_async`

| 参数                   | 类型      | 默认值        | 描述                               |
|----------------------|---------|------------|----------------------------------|
| `text`               | `str`   | —          | 待合成文本                            |
| `name`               | `str`   | `female`   | 发音人角色，可以通过`add_speaker`接口添加自定义角色 |
| `pitch`              | `str`   | `moderate` | 模型不支持该功能                         |
| `speed`              | 同上      | `moderate` | 模型不支持该功能                         |
| `temperature`        | `float` | `0.9`      | 采样温度                             |
| `top_k`              | `int`   | `50`       | top-k 采样                         |
| `top_p`              | `float` | `0.95`     | top-p 采样                         |
| `repetition_penalty` | `float` | `1.0`      | 重复惩罚系数                           |
| `max_tokens`         | `int`   | `4096`     | LLM 最大生成 token 数                 |
| `length_threshold`   | `int`   | `50`       | 文本切分阈值，超过则按 `window_size` 分段     |
| `window_size`        | `int`   | `50`       | 文本滑动窗口大小                         |
| `time_step`          | `int`   | `32`       | 扩散变压器的推理步骤                       |
| `p_w`                | `float` | `1.6`      | 清晰度权重                            | 
| `t_w`                | `float` | `2.5`      | 相似性权重                            |

#### `clone_voice_async`

| 参数                | 类型                      | 默认值    | 描述                                                                     |
|-------------------|-------------------------|--------|------------------------------------------------------------------------|
| `text`            | `str`                   | —      | 待克隆文本                                                                  |
| `reference_audio` | `tuple[str\|bytes,str]` | —      | 参考音频元组，第一个是参考音频的路径或bytes，第二个是WaveVAE提取后的特征(npy文件)。npy文件需要联系MegaTTS官方获取 | 参考音频路径或字节数据                    |
| `reference_text`  | `str`                   | `None` | 该模型不需要这个参数                                                             |
| 其余同 `speak_async` |                         |        | 如 `temperature`, `pitch`, `speed` 等                                    |

