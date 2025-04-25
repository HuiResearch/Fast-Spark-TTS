## MegaTTS3 推理使用文档

> 本文档介绍如何使用 MegaTTS3 进行语音克隆。

---

### 目录

1. [快速开始](#快速开始)
    - [语音克隆](#语音克隆)
    - [内置角色合成](#内置角色合成)
    - [多角色合成](#多角色合成)

2. [API 参考](#api-参考)

---

## 快速开始

以下示例基于 `AsyncMega3Engine` 接口：

### 初始化引擎

```python
from flashtts import AsyncMega3Engine

engine = AsyncMega3Engine(
    model_path="checkpoints/MegaTTS3",
    llm_device="cuda",
    tokenizer_device="cuda",
    backend="vllm",  # 可替换为 torch、vllm、sglang、llama-cpp、mlx-lm
    torch_dtype="float16"  # mega tts支持任意精度
)
```

### 语音克隆

mega tts模型的参考音频是一个元组，需要传入wav文件和编码后的npy文件。

出于安全考虑，`MegaTTS3`
团队未上传WaveVAE编码器参数。因此，只能从此处链接下载参考音频进行推理：[参考音频](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr)
。如果需要使用自己的音频，请参考MegaTTS3项目的说明上传音频文件，请求官方处理：[MegaTTS3](https://github.com/bytedance/MegaTTS3/tree/main?tab=readme-ov-file#inference)

```python
if __name__ == '__main__':
    wav = asyncio.run(
        engine.clone_voice_async(
            text="另一边的桌上,一位读书人嗤之以鼻道,'佛子三藏,神子燕小鱼是什么样的人物,李家的那个李子夜如何与他们相提并论？",
            reference_audio=("data/mega-roles/蔡徐坤/蔡徐坤1.wav", "data/mega-roles/蔡徐坤/蔡徐坤1.npy"),
            max_tokens=512
        )
    )
    engine.write_audio(wav, "cloned.wav")
```

### 内置角色合成

使用`add_speaker`接口添加角色，通过传入`name`指定已有角色进行合成。

```python
# 添加自定义角色
import asyncio


async def run():
    await engine.add_speaker(
        name="太乙真人",
        audio=(
            "data/mega-roles/太乙真人/太乙真人.wav",
            "data/mega-roles/太乙真人/太乙真人.npy"
        )
    )

    # 合成角色音频
    text = "..."
    wav = await engine.speak_async(text=text, name="太乙真人")
    engine.write_audio(wav, "role.wav")


if __name__ == '__main__':
    asyncio.run(run())

```

### 多角色合成

使用`add_speaker`接口添加角色，通过角色标识符`<role:角色名>`指示每个句子所属的角色。

```python
# 添加自定义角色
import asyncio


async def run():
    await engine.add_speaker(
        name="太乙真人",
        audio=(
            "data/mega-roles/太乙真人/太乙真人.wav",
            "data/mega-roles/太乙真人/太乙真人.npy"
        )
    )
    await engine.add_speaker(
        name="御姐",
        audio=(
            "data/mega-roles/御姐/御姐配音.wav",
            "data/mega-roles/御姐/御姐配音.npy"
        )
    )
    # 合成多角色对话
    multi_text = "<role:太乙真人>...<role:御姐>..."
    wav = await engine.multi_speak_async(multi_text)
    engine.write_audio(wav, "multi.wav")


if __name__ == '__main__':
    asyncio.run(run())

```

## API 参考

详见 [fast_tts/engine/mega_engine](../../../flashtts/engine/mega_engine.py) 下各方法注释，主要类和方法：

| 类 / 方法              | 描述                                    |
|---------------------|---------------------------------------|
| `AsyncMega3Engine`  | 封装`MegaTTS3`初始化，MegaTTS不支持流式接口。       |
| `add_speaker`       | 添加内置音频角色                              |
| `delete_speaker`    | 删除内置角色                                |
| `list_roles`        | 查看已有角色列表                              |
| `speak_async`       | 异步文本合成，返回音频数据(np.int16)，需要添加音频角色后才可使用 |
| `clone_voice_async` | 异步语音克隆，基于参考音频生成语音                     |
| `multi_speak_async` | 多角色语音合成，在父类`BaseEngine`中定义            |

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

#### 1. `add_speaker`：添加角色

| 参数               | 类型                      | 默认值    | 描述                                                                     |
|------------------|-------------------------|--------|------------------------------------------------------------------------|
| `name`           | `str`                   | —      | 添加的角色命名                                                                |
| `audio`          | `tuple[str\|bytes,str]` | —      | 参考音频元组，第一个是参考音频的路径或bytes，第二个是WaveVAE提取后的特征(npy文件)。npy文件需要联系MegaTTS官方获取 |
| `reference_text` | `str`                   | `None` | MegaTTS模型不需要这个参数                                                       |

#### 2. `delete_speaker`：删除角色

| 参数     | 类型    | 默认值 | 描述      |
|--------|-------|-----|---------|
| `name` | `str` | —   | 待删除的角色名 |

#### 3. `list_roles`查看已有角色列表

返回`list[str]`

#### 4. `speak_async`

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

#### 5. `clone_voice_async`

| 参数                | 类型                      | 默认值    | 描述                                                                     |
|-------------------|-------------------------|--------|------------------------------------------------------------------------|
| `text`            | `str`                   | —      | 待克隆文本                                                                  |
| `reference_audio` | `tuple[str\|bytes,str]` | —      | 参考音频元组，第一个是参考音频的路径或bytes，第二个是WaveVAE提取后的特征(npy文件)。npy文件需要联系MegaTTS官方获取 | 
| `reference_text`  | `str`                   | `None` | 该模型不需要这个参数                                                             |
| 其余同 `speak_async` |                         |        | 如 `temperature`, `pitch`, `speed` 等                                    |

#### 6. `multi_speak_async`：多角色音频合成

| 参数                | 类型    | 默认值 | 描述                                      |
|-------------------|-------|-----|-----------------------------------------|
| `text`            | `str` | —   | 多角色对话文本，通过角色标识符`<role:角色名>`指示每个句子所属的角色。 |
| 其他同 `speak_async` | —     | —   | 包括 `temperature`、`top_k`、`top_p` 等      |

#### 7. `multi_speak_stream_async`：多角色流式音频合成

| 参数                       | 类型    | 默认值 | 描述                                      |
|--------------------------|-------|-----|-----------------------------------------|
| `text`                   | `str` | —   | 多角色对话文本，通过角色标识符`<role:角色名>`指示每个句子所属的角色。 |
| 其他同 `speak_stream_async` | —     | —   | 包括 `temperature`、`top_k`、`top_p` 等      |
