## Spark-TTS Inference Usage Guide

> This document explains how to use Spark-TTS for speech synthesis.

---

### Table of Contents

1. [Quick Start](#quick-start)
    - [Initialize Engine](#initialize-engine)
    - [Single Sentence Synthesis](#single-sentence-synthesis)
    - [Long Text Synthesis](#long-text-synthesis)
    - [Multi-Speaker Synthesis](#multi-speaker-synthesis)
    - [Voice Cloning](#voice-cloning)
2. [Advanced Features](#advanced-features)
    - [Acoustic Token Reuse](#acoustic-token-reuse)
    - [Streaming Output](#streaming-output)
3. [API Reference](#api-reference)

---

## Quick Start

The following examples are based on the `AsyncSparkEngine` interface:

### Initialize Engine

```python
from flashtts import AsyncSparkEngine

engine = AsyncSparkEngine(
    model_path="checkpoints/Spark-TTS-0.5B",
    llm_device="cuda",
    tokenizer_device="cuda",
    detokenizer_device="cuda",
    backend="vllm",  # Can be torch, vllm, sglang, llama-cpp, mlx-lm
    torch_dtype="bfloat16"  # Note: Spark-TTS does not support float16, use float32 or bfloat16
)
```

### Single Sentence Synthesis

```python
import asyncio

if __name__ == '__main__':
    text = "手握日月摘星辰，世间无我这般人。"
    # 异步调用示例
    wav = asyncio.run(engine.speak_async(text=text, name="female"))
    # 保存音频
    engine.write_audio(wav, "output.wav")
```

### Long Text Synthesis

```python
import asyncio

if __name__ == '__main__':
    long_text = "..."  # Long text over 200 characters
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

### Multi-Speaker Synthesis

Use `<role:SpeakerName>` to indicate the speaker for each segment.

```python
import asyncio


async def run():
    await engine.add_speaker("哪吒", audio="data/roles/哪吒.wav")
    await engine.add_speaker("李靖", audio="data/roles/李靖.wav")
    multi_text = "<role:哪吒>...<role:李靖>..."
    wav = await engine.multi_speak_async(multi_text)
    engine.write_audio(wav, "multi.wav")


if __name__ == '__main__':
    asyncio.run(run())

```

You can also set `pitch` and `speed` for each speaker. Valid values:  
`very_low`, `low`, `moderate`, `high`, `very_high`.

```python
import asyncio


async def run():
    await engine.add_speaker("哪吒", audio="data/roles/哪吒.wav")
    await engine.add_speaker("李靖", audio="data/roles/李靖.wav")

    multi_text = "<role:哪吒,pitch:very_high,speed:very_high>...<role:李靖>..."
    wav = await engine.multi_speak_async(multi_text)
    engine.write_audio(wav, "multi.wav")


if __name__ == '__main__':
    asyncio.run(run())

```

### Voice Cloning

Currently supports only `.wav` format, preferably under 30 seconds.

```python
if __name__ == '__main__':
    wav = asyncio.run(
        engine.clone_voice_async(
            text="Text to be cloned",
            reference_audio="data/roles/余承东/reference_audio.wav",
            pitch="high",
            speed="moderate"
        )
    )
    engine.write_audio(wav, "cloned.wav")
```

---

## Advanced Features

### Acoustic Token Reuse

1. Generate and save acoustic tokens:
   ```python
   wav, tokens = asyncio.run(
       engine.speak_async(..., return_acoustic_tokens=True)
   )
   tokens.save("acoustic_tokens.txt")
   ```
2. Reuse saved tokens:
   ```python
   from flashtts import SparkAcousticTokens
   tokens = SparkAcousticTokens.load("acoustic_tokens.txt")
   wav2 = asyncio.run(
       engine.speak_async(..., acoustic_tokens=tokens)
   )
   ```

### Streaming Output

All methods support streaming. Example for `speak_async`:

```python
chunks = []
async for chunk in engine.speak_stream_async(text, name="female"):
    chunks.append(chunk)
audio = np.concatenate(chunks)
engine.write_audio(audio, "stream.wav")
```

---

## API Reference

See method annotations in [fast_tts/engine/spark_engine](../../../flashtts/engine/spark_engine.py).  
Key classes and methods:

| Class / Method             | Description                               |
|----------------------------|-------------------------------------------|
| `AsyncSparkEngine`         | Initializes Spark-TTS                     |
| `add_speaker`              | Add custom speaker                        |
| `delete_speaker`           | Remove speaker                            |
| `list_roles`               | List all speakers                         |
| `speak_async`              | Async text-to-speech, returns audio array |
| `speak_stream_async`       | Async streaming TTS                       |
| `clone_voice_async`        | Async voice cloning                       |
| `clone_voice_stream_async` | Streaming voice cloning                   |
| `multi_speak_async`        | Multi-speaker synthesis                   |
| `multi_speak_stream_async` | Streaming multi-speaker synthesis         |

### `AsyncSparkEngine` Init Parameters

| Parameter                     | Type    | Default | Description                                                                                                                                       |
|-------------------------------|---------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `model_path`                  | `str`   | —       | Root model directory containing LLM, tokenizer, and detokenizer weights                                                                           |
| `max_length`                  | `int`   | `32768` | Max LLM context length                                                                                                                            |
| `llm_device`                  | `str`   | `auto`  | LLM device                                                                                                                                        |
| `tokenizer_device`            | —       | `auto`  | Tokenizer device                                                                                                                                  |
| `detokenizer_device`          | —       | `auto`  | Detokenizer device                                                                                                                                |
| `backend`                     | `str`   | `torch` | Backend: `torch`, `vllm`, `sglang`, `llama-cpp`, `mlx-lm`, `tensorrt-llm`                                                                         |
| `--llm_tensorrt_path`         | `str`   | `None`  | Path to the TensorRT model. Only effective when the backend is set to `tensorrt-llm`. If not provided, defaults to `{model_path}/tensorrt-engine` |
| `wav2vec_attn_implementation` | `str`   | `eager` | `wav2vec` attention backend: `sdpa`, `flash_attention_2`, or `eager`                                                                              |
| `llm_attn_implementation`     | —       | `eager` | LLM attention implementation                                                                                                                      |
| `torch_dtype`                 | `str`   | `auto`  | LLM weight dtype: `float16`, `bfloat16`, or `float32`                                                                                             |
| `llm_gpu_memory_utilization`  | `float` | `0.6`   | GPU memory utilization limit (vllm/sglang only)                                                                                                   |
| `batch_size`                  | `int`   | `1`     | Tokenizer / detokenizer batch size                                                                                                                |
| `llm_batch_size`              | `int`   | `256`   | LLM decode batch size                                                                                                                             |
| `wait_timeout`                | `float` | `0.01`  | Async wait timeout                                                                                                                                |
| `seed`                        | `int`   | `0`     | Random seed                                                                                                                                       |

---

### 1. `add_speaker` – Add Speaker

| Parameter        | Type             | Default | Description                                                   |
|------------------|------------------|---------|---------------------------------------------------------------|
| `name`           | `str`            | —       | Name of the speaker to add                                    |
| `audio`          | `bytes` \| `str` | —       | Reference audio file path or audio bytes                      |
| `reference_text` | `str`            | `None`  | Transcript of the reference audio; omit for low-quality audio |

---

### 2. `delete_speaker` – Delete Speaker

| Parameter | Type  | Default | Description               |
|-----------|-------|---------|---------------------------|
| `name`    | `str` | —       | Name of speaker to delete |

---

### 3. `list_roles` – List Speakers

Returns a `list[str]` containing all current speaker names.

---

### 4. `speak_async` – Synthesize with a Single Voice

| Parameter                | Type                           | Default      | Description                                                  |
|--------------------------|--------------------------------|--------------|--------------------------------------------------------------|
| `text`                   | `str`                          | —            | Text to synthesize                                           |
| `name`                   | `str`                          | `"female"`   | Speaker name (built-in: `female`, `male`, or custom)         |
| `pitch`                  | `str`                          | `"moderate"` | Pitch: `very_low`, `low`, `moderate`, `high`, `very_high`    |
| `speed`                  | `str`                          | `"moderate"` | Speed: same options as `pitch`                               |
| `temperature`            | `float`                        | `0.9`        | Sampling temperature                                         |
| `top_k`                  | `int`                          | `50`         | Top-k sampling                                               |
| `top_p`                  | `float`                        | `0.95`       | Top-p sampling                                               |
| `repetition_penalty`     | `float`                        | `1.0`        | Penalty to reduce repetition                                 |
| `max_tokens`             | `int`                          | `4096`       | Max tokens to generate                                       |
| `length_threshold`       | `int`                          | `50`         | If text length exceeds, it will be split using `window_size` |
| `window_size`            | `int`                          | `50`         | Text chunk size when splitting                               |
| `split_fn`               | `Callable[[str], list[str]]`   | `None`       | Custom text splitting function                               |
| `acoustic_tokens`        | `SparkAcousticTokens` \| `str` | `None`       | Use pre-generated acoustic tokens (only for `female`/`male`) |
| `return_acoustic_tokens` | `bool`                         | `False`      | Whether to return acoustic tokens (only for `female`/`male`) |

---

### 5. `speak_stream_async` – Streaming Single-Voice Synthesis

Same parameters as `speak_async`, plus:

| Parameter                       | Type    | Default | Description                                         |
|---------------------------------|---------|---------|-----------------------------------------------------|
| `audio_chunk_duration`          | `float` | `1.0`   | Duration (sec) of each audio chunk                  |
| `max_audio_chunk_duration`      | `float` | `8.0`   | Max duration per audio chunk                        |
| `audio_chunk_size_scale_factor` | `float` | `2.0`   | Chunk duration scale factor                         |
| `audio_chunk_overlap_duration`  | `float` | `0.1`   | Overlap (fade-in/out) duration between chunks (sec) |

---

### 6. `clone_voice_async` – Voice Cloning

| Parameter         | Type             | Default | Description                                                       |
|-------------------|------------------|---------|-------------------------------------------------------------------|
| `text`            | `str`            | —       | Text to synthesize                                                |
| `reference_audio` | `bytes` \| `str` | —       | Reference audio file or byte data                                 |
| `reference_text`  | `str`            | `None`  | Optional transcript of reference audio                            |
| Others            | —                | —       | Inherits `pitch`, `speed`, `temperature`, etc. from `speak_async` |

---

### 7. `clone_voice_stream_async` – Streaming Voice Cloning

Same as `clone_voice_async`, plus streaming-specific chunk parameters:

| Parameter                       | Type                         | Default | Description                            |
|---------------------------------|------------------------------|---------|----------------------------------------|
| `text`                          | `str`                        | —       | Text to synthesize                     |
| `reference_audio`               | `bytes` \| `str`             | —       | Reference audio                        |
| `reference_text`                | `str`                        | `None`  | Optional transcript of reference audio |
| `split_fn`                      | `Callable[[str], list[str]]` | `None`  | Custom text split function             |
| `audio_chunk_duration`          | `float`                      | `1.0`   | Duration per audio chunk               |
| `max_audio_chunk_duration`      | `float`                      | `8.0`   | Max chunk duration                     |
| `audio_chunk_size_scale_factor` | `float`                      | `2.0`   | Chunk duration scaling                 |
| `audio_chunk_overlap_duration`  | `float`                      | `0.1`   | Overlap (fade-in/out) duration         |

---

### 8. `multi_speak_async` – Multi-Speaker Synthesis

| Parameter | Type  | Default | Description                                                        |
|-----------|-------|---------|--------------------------------------------------------------------|
| `text`    | `str` | —       | Dialog with `<role:SpeakerName>` tags to assign speakers           |
| Others    | —     | —       | Inherits from `speak_async`: `temperature`, `top_k`, `top_p`, etc. |

---

### 9. `multi_speak_stream_async` – Streaming Multi-Speaker Synthesis

| Parameter | Type  | Default | Description                                                                  |
|-----------|-------|---------|------------------------------------------------------------------------------|
| `text`    | `str` | —       | Dialog with `<role:SpeakerName>` tags to assign speakers                     |
| Others    | —     | —       | Inherits from `speak_stream_async`: streaming parameters, pitch, speed, etc. |
