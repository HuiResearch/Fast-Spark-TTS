# AutoEngine 使用文档

> 本文档介绍如何使用 `AutoEngine` 根据模型目录自动选择 SparkTTS、OrpheusTTS 或 MegaTTS 引擎，统一完成语音合成、语音克隆和多角色对话等任务。

---

## 快速开始

```python
import asyncio
from flashtts import AutoEngine

# 假设 checkpoints/YourModelDir 为 Spark、Orpheus 或 Mega 对应的模型目录
engine = AutoEngine(
    model_path="checkpoints/YourModelDir",
    snac_path=None,  # 当使用 Orpheus 且目录中无 snac 子目录时需指定
    lang="mandarin",  # 仅对 Orpheus 生效
    llm_device="cuda",  # LLM 计算设备
    tokenizer_device="cuda",  # 音频 tokenizer 设备（Spark/Mega）
    detokenizer_device="cuda",  # 音频 detokenizer 设备（Spark/Orpheus）
    backend="vllm",  # LLM 加速后端
    torch_dtype="float32",  # LLM 权重量化类型
    batch_size=1,
    llm_batch_size=256,
    wait_timeout=0.01,
    seed=42
)


async def main():
    # 单句合成（示例以 Spark 为例，可切换到 Orpheus/Mega）
    wav = await engine.speak_async(
        text="你好，世界！",
        name="female",  # Spark 默认内置角色 “female”
        pitch="moderate",  # 仅 Spark
        speed="moderate",  # 仅 Spark
        temperature=0.9,
        top_k=50,
        top_p=0.95
    )
    engine.write_audio(wav, "output.wav")


if __name__ == '__main__':
    asyncio.run(main())
```

---

## 引擎自动检测逻辑

`AutoEngine` 会根据 `model_path` 下的子目录名称自动判断使用哪种引擎：

- **Spark**：同时包含 `LLM`、`BiCodec` 与 `wav2vec2-large-xlsr-53` 子目录时，选用 Spark 引擎 。
- **Mega**：同时包含 `aligner_lm`、`diffusion_transformer`、`duration_lm`、`g2p` 与 `wavvae` 子目录时，选用 Mega 引擎 。
- **Orpheus**：包含 `snac` 子目录，或外部通过 `snac_path` 指定时，选用 Orpheus 引擎。
- 否则抛出 `RuntimeError: No engine found`。

---

## 初始化参数

| 参数                            | 类型                                                      | 默认值       | 描述                                                    |
|-------------------------------|---------------------------------------------------------|-----------|-------------------------------------------------------|
| `model_path`                  | `str`                                                   | —         | 模型根目录路径，包含引擎对应的子目录。                                   |
| `max_length`                  | `int`                                                   | `32768`   | LLM 最大上下文长度，与 Spark 引擎一致；Orpheus 默认为 `8192`。          |
| `snac_path`                   | `Optional[str]`                                         | `None`    | Orpheus SNAC 模块权重路径；若 `model_path` 下已有 `snac` 子目录可省略。 |
| `lang`                        | `Literal["mandarin", "french", …, "english", None]`     | `None`    | Orpheus 支持的语言类型。                                      |
| `llm_device`                  | `Literal["cpu","cuda","mps","auto"] \| str`             | `"auto"`  | LLM 模块计算设备。                                           |
| `tokenizer_device`            | 同上                                                      | `"auto"`  | 音频 tokenizer 计算设备（Spark/Mega）。                        |
| `detokenizer_device`          | 同上                                                      | `"auto"`  | 音频 detokenizer 计算设备（Spark/Orpheus）。                   |
| `backend`                     | `Literal["vllm","llama-cpp","sglang","torch","mlx-lm"]` | `"torch"` | LLM 加速后端，各引擎均支持。                                      |
| `wav2vec_attn_implementation` | `Optional[Literal["sdpa","flash_attention_2","eager"]]` | `None`    | Spark wav2vec 注意力算子实现。                                |
| `llm_attn_implementation`     | 同上                                                      | `None`    | LLM 注意力算子实现。                                          |
| `torch_dtype`                 | `Literal['float16','bfloat16','float32','auto']`        | `"auto"`  | LLM 权重量化类型；Mega 支持任意精度。                               |
| `llm_gpu_memory_utilization`  | `Optional[float]`                                       | `0.6`     | vllm/sglang 后端显存占用率上限。                                |
| `cache_implementation`        | `Optional[str]`                                         | `None`    | 缓存策略实现名称，仅在`backend`为`torch`时生效。                      |
| `batch_size`                  | `int`                                                   | `1`       | tokenizer / detokenizer 并发批处理大小。                      |
| `llm_batch_size`              | `int`                                                   | `256`     | LLM 解码并行批处理大小。                                        |
| `wait_timeout`                | `float`                                                 | `0.01`    | tokenizer / detokenizer 异步等待超时。                       |
| `seed`                        | `int`                                                   | `0`       | 随机种子。                                                 |
| `**kwargs`                    |                                                         | —         | LLM后端特定额外参数                                           |

初始化时会打印类似：

```
Initializing `AutoEngine(engine=spark)` with config: (model_path='…', max_length=32768, …)
```

并将 `self._engine` 设为对应的 `AsyncSparkEngine`、`AsyncOrpheusEngine` 或 `AsyncMega3Engine` 实例。

---

## 通用方法

所有引擎均支持以下统一接口：

#### `write_audio(audio: np.ndarray, filepath: str)`

将 `audio` 写入指定文件（WAV 格式）。

#### `list_roles() → list[str]`

返回当前引擎中可用的角色列表。

- Spark: 查看已添加的角色
- Orpheus: 查看内置角色列表
- Mega: 查看已添加的角色

#### `add_speaker(name: str, audio, reference_text: Optional[str]=None) → Awaitable`

为 Spark/Orpheus/Mega 添加自定义角色：

- **Spark**: `audio` 接受 `bytes` 或路径字符串，`reference_text` 为转录文本（可不填写）。
- **Orpheus**: 不支持该功能 。
- **Mega**: `audio` 为 `(wav_path, npy_path)` 二元组。

#### `delete_speaker(name: str) → Awaitable`

删除指定角色。

---

## 核心合成与克隆接口

| 方法                                                          | 描述                               |
|-------------------------------------------------------------|----------------------------------|
| `speak_async(...) → np.ndarray`                             | 异步文本合成，返回完整音频（所有引擎）              |
| `speak_stream_async(...) → AsyncIterator[np.ndarray]`       | 异步流式合成，按块返回音频（Spark/Orpheus）     |
| `clone_voice_async(...) → np.ndarray`                       | 异步语音克隆，基于参考音频生成新语音（Spark/Mega）   |
| `clone_voice_stream_async(...) → AsyncIterator[np.ndarray]` | 异步流式克隆（Spark）                    |
| `multi_speak_async(...) → np.ndarray`                       | 多角色对话合成，使用 `<role:角色名>` 标记（所有引擎） |
| `multi_speak_stream_async(...) → AsyncIterator[np.ndarray]` | 多角色流式合成（Spark/Orpheus）           |

> **参数说明（以 `speak_async` 为例）**
> - `text: str`：待合成文本
> - `name: Optional[str]`：发音人角色；Spark 默认 `"female"`
> - `pitch, speed: Literal["very_low","low","moderate","high","very_high"]`：仅 Spark 支持
> - `temperature: float = 0.9`：采样温度
> - `top_k: int = 50`；`top_p: float = 0.95`；`repetition_penalty: float = 1.0`：解码采样参数
> - `max_tokens: int = 4096`：LLM 最大生成 token 数
> - `length_threshold: int = 50`；`window_size: int = 50`：长文本分段阈值与滑动窗口大小
> - `split_fn: Optional[Callable[[str], list[str]]]`：自定义分段函数

其他方法参数与上述保持一致，详情请参阅对应引擎文档。

---

## 注意事项

1. **引擎差异**：
    - **Spark** 支持音色复用（`return_acoustic_tokens`、`acoustic_tokens`）及完整流式接口。
    - **Orpheus** 不支持语音克隆，强调长文本和多种情感标签。
    - **Mega** 需要提供 WaveVAE 特征文件进行克隆，支持多角色合成。

2. **性能调优**：根据硬件环境调整 `llm_device`、`backend`、`torch_dtype` 及显存利用率参数，可显著影响速度与质量。

---

使用 `AutoEngine`，即可实现对多种底层 TTS 引擎的无缝切换与管理，大幅简化模型部署与调用复杂度。