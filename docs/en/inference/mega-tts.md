## MegaTTS3 Inference Guide

> This guide explains how to use **MegaTTS3** for voice cloning.

---

### Table of Contents

1. [Quick Start](#quick-start)
    - [Voice Cloning](#voice-cloning)
    - [Built-in Speaker Synthesis](#built-in-speaker-synthesis)
    - [Multi-Speaker Synthesis](#multi-speaker-synthesis)

2. [API Reference](#api-reference)

---

## Quick Start

All examples below use the `AsyncMega3Engine` interface:

### Initialize the Engine

```python
from flashtts import AsyncMega3Engine

engine = AsyncMega3Engine(
    model_path="checkpoints/MegaTTS3",
    llm_device="cuda",
    tokenizer_device="cuda",
    backend="vllm",  # Options: torch, vllm, sglang, llama-cpp, mlx-lm
    torch_dtype="float16"  # MegaTTS supports all precision types
)
```

### Voice Cloning

MegaTTS models require a **tuple of audio**: a `.wav` file and an encoded `.npy` file as the reference.

> ⚠️ For security reasons, the MegaTTS3 team does not provide the WaveVAE encoder parameters.  
> For inference, use this shared folder to download preprocessed reference audio:  
> [Reference Audio](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr)
>
> To use your own audio, follow the MegaTTS3 project instructions to request official encoding:  
> [MegaTTS3 GitHub](https://github.com/bytedance/MegaTTS3/tree/main?tab=readme-ov-file#inference)

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

### Built-in Speaker Synthesis

Use the `add_speaker` method to register custom roles, then use `name` to synthesize speech.

```python
import asyncio


async def run():
    await engine.add_speaker(
        name="太乙真人",
        audio=(
            "data/mega-roles/太乙真人/太乙真人.wav",
            "data/mega-roles/太乙真人/太乙真人.npy"
        )
    )

    text = "..."
    wav = await engine.speak_async(text=text, name="太乙真人")
    engine.write_audio(wav, "role.wav")


if __name__ == '__main__':
    asyncio.run(run())
```

### Multi-Speaker Synthesis

Define speakers with `add_speaker`, then indicate each line’s speaker using `<role:SpeakerName>` tags.

```python
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

---

## API Reference

Refer to inline comments in  
[`fast_tts/engine/mega_engine.py`](../../../flashtts/engine/mega_engine.py) for full documentation.  
Key classes and methods:

| Class / Method      | Description                                                          |
|---------------------|----------------------------------------------------------------------|
| `AsyncMega3Engine`  | Wrapper for initializing MegaTTS3 (note: streaming is not supported) |
| `add_speaker`       | Register a built-in audio role                                       |
| `delete_speaker`    | Remove a registered role                                             |
| `list_roles`        | List currently registered roles                                      |
| `speak_async`       | Async TTS, returns full audio (must add roles first)                 |
| `clone_voice_async` | Async voice cloning with reference audio                             |
| `multi_speak_async` | Multi-speaker synthesis (inherited from `BaseEngine`)                |

### `AsyncMega3Engine` Initialization Parameters

| Parameter                    | Type    | Default   | Description                                               |
|------------------------------|---------|-----------|-----------------------------------------------------------|
| `model_path`                 | `str`   | —         | Path to the model root directory                          |
| `max_length`                 | `int`   | `32768`   | Max context length for LLM                                |
| `llm_device`                 | `str`   | `"auto"`  | LLM computation device                                    |
| `tokenizer_device`           | `str`   | `"auto"`  | Tokenizer device                                          |
| `backend`                    | `str`   | `"torch"` | Backend: `torch`, `vllm`, `sglang`, `llama-cpp`, `mlx-lm` |
| `llm_attn_implementation`    | `str`   | `"eager"` | LLM attention implementation                              |
| `torch_dtype`                | `str`   | `"auto"`  | LLM weight precision (`float16`, `bfloat16`, `float32`)   |
| `llm_gpu_memory_utilization` | `float` | `0.6`     | Max GPU memory utilization (vllm/sglang only)             |
| `batch_size`                 | `int`   | `1`       | Batch size for tokenizer/detokenizer                      |
| `llm_batch_size`             | `int`   | `256`     | Batch size for LLM decoding                               |
| `wait_timeout`               | `float` | `0.01`    | Async wait timeout for tokenizer/detokenizer              |
| `seed`                       | `int`   | `0`       | Random seed                                               |

### Main API Parameter Reference

#### 1. `add_speaker`

| Parameter        | Type                       | Default | Description                                                                   |
|------------------|----------------------------|---------|-------------------------------------------------------------------------------|
| `name`           | `str`                      | —       | Name for the speaker                                                          |
| `audio`          | `tuple[str \| bytes, str]` | —       | Tuple of reference `.wav` and `.npy` file. `.npy` must be provided by MegaTTS |
| `reference_text` | `str`                      | `None`  | Not required for MegaTTS                                                      |

#### 2. `delete_speaker`

| Parameter | Type  | Default | Description            |
|-----------|-------|---------|------------------------|
| `name`    | `str` | —       | Speaker name to delete |

#### 3. `list_roles`

Returns: `list[str]`

#### 4. `speak_async`

| Parameter            | Type    | Default      | Description                                   |
|----------------------|---------|--------------|-----------------------------------------------|
| `text`               | `str`   | —            | Text to synthesize                            |
| `name`               | `str`   | `"female"`   | Speaker name (must be registered)             |
| `pitch`              | `str`   | `"moderate"` | Not supported                                 |
| `speed`              | `str`   | `"moderate"` | Not supported                                 |
| `temperature`        | `float` | `0.9`        | Sampling temperature                          |
| `top_k`              | `int`   | `50`         | Top-k sampling                                |
| `top_p`              | `float` | `0.95`       | Top-p sampling                                |
| `repetition_penalty` | `float` | `1.0`        | Repetition penalty                            |
| `max_tokens`         | `int`   | `4096`       | Max token output from LLM                     |
| `length_threshold`   | `int`   | `50`         | Split threshold for long text                 |
| `window_size`        | `int`   | `50`         | Sliding window size for text                  |
| `time_step`          | `int`   | `32`         | Inference steps for the diffusion transformer |
| `p_w`                | `float` | `1.6`        | Clarity weight                                |
| `t_w`                | `float` | `2.5`        | Similarity weight                             |

#### 5. `clone_voice_async`

| Parameter         | Type                       | Default | Description                                          |
|-------------------|----------------------------|---------|------------------------------------------------------|
| `text`            | `str`                      | —       | Text to synthesize                                   |
| `reference_audio` | `tuple[str \| bytes, str]` | —       | Tuple of `.wav` file and WaveVAE `.npy` feature file |
| `reference_text`  | `str`                      | `None`  | Not used                                             |
| Other Params      | Same as `speak_async`      |         | e.g., `temperature`, `pitch`, `speed`, etc.          |

#### 6. `multi_speak_async`

| Parameter | Type  | Default | Description                                                  |
|-----------|-------|---------|--------------------------------------------------------------|
| `text`    | `str` | —       | Multi-speaker dialogue using `<role:Name>` tags              |
| Others    | —     | —       | Same as `speak_async`: `temperature`, `top_k`, `top_p`, etc. |

#### 7. `multi_speak_stream_async`

| Parameter | Type  | Default | Description                                        |
|-----------|-------|---------|----------------------------------------------------|
| `text`    | `str` | —       | Multi-speaker dialogue using `<role:Name>` tags    |
| Others    | —     | —       | Same as `speak_stream_async`: streaming parameters |