## Orpheus-TTS Inference Usage Guide

> This document provides instructions for using Orpheus-TTS for speech synthesis.

---

### Table of Contents

1. [Quick Start](#quick-start)
    - [Initialize the Engine](#Initialize-the-Engine)
    - [Single Sentence Synthesis](#single-sentence-synthesis)
    - [Long Text Synthesis](#long-text-synthesis)
    - [Multi-Speaker Synthesis](#multi-speaker-synthesis)
2. [Advanced Features](#advanced-features)
    - [Streaming Output](#streaming-output)
3. [API Reference](#api-reference)

---

## Quick Start

The following examples are based on the `AsyncOrpheusEngine` interface:

### Initialize the Engine

```python
from flashtts import AsyncOrpheusEngine


def prepare_engine():
    engine = AsyncOrpheusEngine(
        model_path="checkpoints/orpheus-3b-0.1-ft",
        snac_path="checkpoints/snac_24khz",
        max_length=8192,
        llm_device="cuda",
        detokenizer_device="cuda",
        backend="vllm"
    )
    return engine
```

### Single Sentence Synthesis

`Orpheus-TTS` only supports text-to-speech synthesis. Voice cloning is not supported, and cloning interfaces are not
provided.

The model supports emotion tags such as `<laugh>`, `<cough>`, etc. For supported tags per model, refer to:  
https://canopylabs.ai/releases/orpheus_can_speak_any_language#info

```python
import asyncio

if __name__ == '__main__':
    text = "I am tara. <laugh>"
    # Asynchronous synthesis example
    wav = asyncio.run(engine.speak_async(text=text, name="tara"))
    # Save audio
    engine.write_audio(wav, "output.wav")
```

### Long Text Synthesis

```python
import asyncio

if __name__ == '__main__':
    long_text = "..."  # >200 characters
    wav = asyncio.run(
        engine.speak_async(
            text=long_text,
            name="tara",
            length_threshold=50,
            window_size=50
        )
    )
    engine.write_audio(wav, "long_output.wav")
```

### Multi-Speaker Synthesis

Use the role tag `<role:RoleName>` to indicate which role is speaking.

Voice cloning and custom role addition are currently not supported.

```python
import asyncio


async def run():
    multi_text = "<role:tara>...<role:dan>..."
    wav = await engine.multi_speak_async(multi_text)
    engine.write_audio(wav, "multi.wav")


asyncio.run(run())
```

---

## Advanced Features

### Streaming Output

All OrpheusTTS interfaces support streaming synthesis:

```python
if __name__ == '__main__':

    chunks = []
    async for chunk in engine.speak_stream_async(text, name="tara"):
        chunks.append(chunk)
    audio = np.concatenate(chunks)
    engine.write_audio(audio, "stream.wav")
```

---

## API Reference

See method docstrings in [fast_tts/engine/orpheus_engine](../../../flashtts/engine/orpheus_engine.py). Main classes and
methods:

| Class / Method             | Description                                      |
|----------------------------|--------------------------------------------------|
| `AsyncOrpheusEngine`       | Initializes the Orpheus-TTS engine               |
| `list_roles`               | Returns a list of available speaker roles        |
| `speak_async`              | Async text-to-speech synthesis                   |
| `speak_stream_async`       | Async streaming synthesis                        |
| `multi_speak_async`        | Multi-speaker synthesis, defined in BaseEngine   |
| `multi_speak_stream_async` | Multi-speaker streaming synthesis, in BaseEngine |

### `AsyncOrpheusEngine` Initialization Parameters

| Parameter                    | Type    | Default | Description                                                                                                                                       |
|------------------------------|---------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `model_path`                 | `str`   | —       | Path to model directory including LLM and SNAC weights                                                                                            |
| `max_length`                 | `int`   | `8192`  | Maximum LLM context length                                                                                                                        |
| `lang`                       | `str`   | `None`  | Language supported by the model                                                                                                                   |
| `snac_path`                  | `str`   | `None`  | Path to SNAC module; optional if included in `model_path`                                                                                         |
| `llm_device`                 | `str`   | `auto`  | Device for LLM module                                                                                                                             |
| `detokenizer_device`         | same    | `auto`  | Device for audio detokenizer                                                                                                                      |
| `backend`                    | `str`   | `torch` | Backend for LLM, supports `torch`, `vllm`, `sglang`, `llama-cpp`, `tensorrt-llm`, `mlx-lm`, etc.                                                  |
| `--llm_tensorrt_path`        | `str`   | `None`  | Path to the TensorRT model. Only effective when the backend is set to `tensorrt-llm`. If not provided, defaults to `{model_path}/tensorrt-engine` |
| `llm_attn_implementation`    | same    | `eager` | Attention implementation (torch only)                                                                                                             |
| `torch_dtype`                | `str`   | `auto`  | Data type for LLM weights, e.g. `float16`, `bfloat16`, `float32`                                                                                  |
| `llm_gpu_memory_utilization` | `float` | `0.6`   | GPU memory cap (only for vllm/sglang)                                                                                                             |
| `batch_size`                 | `int`   | `1`     | Audio tokenizer/detokenizer batch size                                                                                                            |
| `llm_batch_size`             | `int`   | `256`   | LLM decoding batch size                                                                                                                           |
| `wait_timeout`               | `float` | `0.01`  | Timeout for async tokenizer/detokenizer                                                                                                           |
| `seed`                       | `int`   | `0`     | Random seed                                                                                                                                       |

### Main Method Parameter Descriptions

#### 1. `list_roles`: View available roles

Returns `list[str]`

#### 2. `speak_async`

| Parameter            | Type                         | Default    | Description                        |
|----------------------|------------------------------|------------|------------------------------------|
| `text`               | `str`                        | —          | Text to synthesize                 |
| `name`               | `str`                        | `None`     | Speaker role                       |
| `pitch`              | `str`                        | `None`     | Currently unsupported              |
| `speed`              | same                         | `moderate` | Currently unsupported              |
| `temperature`        | `float`                      | `0.9`      | Sampling temperature               |
| `top_k`              | `int`                        | `50`       | Top-k sampling                     |
| `top_p`              | `float`                      | `0.95`     | Top-p sampling                     |
| `repetition_penalty` | `float`                      | `1.0`      | Repetition penalty                 |
| `max_tokens`         | `int`                        | `4096`     | Max tokens for LLM generation      |
| `length_threshold`   | `int`                        | `50`       | Text length threshold for chunking |
| `window_size`        | `int`                        | `50`       | Window size for text sliding       |
| `split_fn`           | `Callable[[str], list[str]]` | `None`     | Custom text splitting function     |

#### 3. `speak_stream_async`: Same parameters as `speak_async`

#### 4. `multi_speak_async`: Multi-speaker synthesis

| Parameter                    | Type  | Default | Description                                    |
|------------------------------|-------|---------|------------------------------------------------|
| `text`                       | `str` | —       | Text with roles marked using `<role:RoleName>` |
| Others same as `speak_async` | —     | —       | Includes `temperature`, `top_k`, `top_p`, etc. |

#### 5. `multi_speak_stream_async`: Streaming multi-speaker synthesis

| Parameter                           | Type  | Default | Description                                    |
|-------------------------------------|-------|---------|------------------------------------------------|
| `text`                              | `str` | —       | Text with roles marked using `<role:RoleName>` |
| Others same as `speak_stream_async` | —     | —       | Includes `temperature`, `top_k`, `top_p`, etc. |
