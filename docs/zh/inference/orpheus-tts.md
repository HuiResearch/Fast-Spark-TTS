## Orpheus-TTS 推理使用文档

> 本文档介绍如何使用 Orpheus-TTS 进行语音合成。

---

### 目录

1. [快速开始](#快速开始)
    - [初始化引擎](#初始化引擎)
    - [单句合成](#单句合成)
    - [长文本合成](#长文本合成)
    - [多角色合成](#多角色合成)
2. [高级特性](#高级特性)
    - [流式输出](#流式输出)
3. [API 参考](#api-参考)

---

## 快速开始

以下示例基于 `AsyncOrpheusEngine` 接口：

### 初始化引擎

```python
from flashtts import AsyncOrpheusEngine


def prepare_engine():
    engine = AsyncOrpheusEngine(
        model_path="checkpoints/3b-zh-ft-research_release",
        snac_path="checkpoints/snac_24khz",
        max_length=8192,
        llm_device="cuda",
        detokenizer_device="cuda",
        backend="vllm"
    )
    return engine
```

### 单句合成

`Orpheus-TTS`仅支持语音合成，克隆效果不佳，未添加克隆的接口。

该模型支持情感标签，比如`<轻笑>`、`<咳嗽>`
等，具体每种模型支持的标签情况参考：https://canopylabs.ai/releases/orpheus_can_speak_any_language#info

```python
import asyncio

if __name__ == '__main__':
    text = "我是长乐啦。 <轻笑>"
    # 异步调用示例
    wav = asyncio.run(engine.speak_async(text=text, name="长乐"))
    # 保存音频
    engine.write_audio(wav, "output.wav")
```

### 长文本合成

```python
import asyncio

if __name__ == '__main__':
    long_text = "..."  # >200 字长文本
    wav = asyncio.run(
        engine.speak_async(
            text=long_text,
            name="长乐",
            length_threshold=50,
            window_size=50
        )
    )
    engine.write_audio(wav, "long_output.wav")
```

### 多角色合成

通过角色标识符`<role:角色名>`指示每个句子所属的角色。

由于不支持语音克隆，暂不支持自定义角色添加。

```python
import asyncio


async def run():
    # 合成多角色对话
    multi_text = "<role:长乐>...<role:白芷>..."
    wav = await engine.multi_speak_async(multi_text)
    engine.write_audio(wav, "multi.wav")


asyncio.run(run())

```

## 高级特性

### 流式输出

`OrpheusTTS`所有接口均支持流式。

```python
if __name__ == '__main__':

    chunks = []
    async for chunk in engine.speak_stream_async(text, name="长乐"):
        chunks.append(chunk)
    audio = np.concatenate(chunks)
    engine.write_audio(audio, "stream.wav")
```

## API 参考

详见 [fast_tts/engine/orpheus_engine](../../../flashtts/engine/orpheus_engine.py) 下各方法注释，主要类和方法：

| 类 / 方法                     | 描述                         |
|----------------------------|----------------------------|
| `AsyncOrpheusEngine`       | 封装`Orpheus-TTS`初始化         |
| `list_roles`               | 查看已有角色列表                   |
| `speak_async`              | 异步文本合成，返回音频数据(np.int16)    |
| `speak_stream_async`       | 异步流式合成，按音频块迭代返回            |
| `multi_speak_async`        | 多角色语音合成，在父类`BaseEngine`中定义 |
| `multi_speak_stream_async` | 多角色语音合成，在父类`BaseEngine`中定义 |

### `AsyncOrpheusEngine` 初始化参数说明

| 参数                           | 类型      | 默认值     | 描述                                                                                |
|------------------------------|---------|---------|-----------------------------------------------------------------------------------|
| `model_path`                 | `str`   | —       | 模型根目录路径，包含 LLM、snac 权重                                                            |
| `max_length`                 | `int`   | `8192`  | LLM 最大上下文长度                                                                       |
| `lang`                       | `str`   | `None`  | 模型支持的语言类型                                                                         |
| `snac_path`                  | `str`   | `None`  | snac模块权重路径，如果`model_path`路径下有命名为`snac`的路径，则无需传入                                   |
| `--llm_tensorrt_path`        | `str`   | `None`  | tensorrt模型路径，仅在backend设置为tensorrt-llm时生效。如果不传入，则默认为`{model_path}/tensorrt-engine` |
| `llm_device`                 | `str`   | `auto`  | LLM模块计算设备                                                                         |
| `detokenizer_device`         | 同上      | `auto`  | 音频 detokenizer 计算设备                                                               |
| `backend`                    | `str`   | `torch` | LLM加速后端，支持`torch`、`vllm`、`sglang`、`llama cpp`、`mlx-lm`、`tensorrt-llm`             |
| `llm_attn_implementation`    | 同上      | `eager` | LLM 注意力算子实现，针对`torch` backend                                                     |
| `torch_dtype`                | `str`   | `auto`  | LLM模块权重量化类型选择，支持 `float16`、`bfloat16`、`float32`                                   |
| `llm_gpu_memory_utilization` | `float` | `0.6`   | 后端显存占用率上限（仅 vllm/sglang 有效）                                                       |
| `batch_size`                 | `int`   | `1`     | 音频 tokenizer / detokenizer 并发批处理大小                                                |
| `llm_batch_size`             | `int`   | `256`   | LLM 解码并行批处理大小                                                                     |
| `wait_timeout`               | `float` | `0.01`  | tokenizer / detokenizer 异步等待超时                                                    |
| `seed`                       | `int`   | `0`     | 随机种子                                                                              |

### 主要接口参数说明

#### 1. `list_roles`查看已有角色列表

返回`list[str]`

#### 2. `speak_async`

| 参数                   | 类型                           | 默认值        | 描述                           |
|----------------------|------------------------------|------------|------------------------------|
| `text`               | `str`                        | —          | 待合成文本                        |
| `name`               | `str`                        | `None`     | 发音人角色                        |
| `pitch`              | `str`                        | `None`     | 暂不支持该参数                      |
| `speed`              | 同上                           | `moderate` | 暂不支持该参数                      |
| `temperature`        | `float`                      | `0.9`      | 采样温度                         |
| `top_k`              | `int`                        | `50`       | top-k 采样                     |
| `top_p`              | `float`                      | `0.95`     | top-p 采样                     |
| `repetition_penalty` | `float`                      | `1.0`      | 重复惩罚系数                       |
| `max_tokens`         | `int`                        | `4096`     | LLM 最大生成 token 数             |
| `length_threshold`   | `int`                        | `50`       | 文本切分阈值，超过则按 `window_size` 分段 |
| `window_size`        | `int`                        | `50`       | 文本滑动窗口大小                     |
| `split_fn`           | `Callable[[str], list[str]]` | `None`     | 自定义分割函数                      |

#### 3. `speak_stream_async` 与 `speak_async` 参数一致。

#### 4. `multi_speak_async`：多角色音频合成

| 参数                | 类型    | 默认值 | 描述                                      |
|-------------------|-------|-----|-----------------------------------------|
| `text`            | `str` | —   | 多角色对话文本，通过角色标识符`<role:角色名>`指示每个句子所属的角色。 |
| 其他同 `speak_async` | —     | —   | 包括 `temperature`、`top_k`、`top_p` 等      |

#### 5. `multi_speak_stream_async`：多角色流式音频合成

| 参数                       | 类型    | 默认值 | 描述                                      |
|--------------------------|-------|-----|-----------------------------------------|
| `text`                   | `str` | —   | 多角色对话文本，通过角色标识符`<role:角色名>`指示每个句子所属的角色。 |
| 其他同 `speak_stream_async` | —     | —   | 包括 `temperature`、`top_k`、`top_p` 等      |
