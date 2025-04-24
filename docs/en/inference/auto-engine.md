# AutoEngine Usage Guide

> This document explains how to use `AutoEngine` to automatically select between SparkTTS, OrpheusTTS, or MegaTTS based
> on the model directory. It unifies tasks such as speech synthesis, voice cloning, and multi-speaker dialogue.

---

## Quick Start

```python
import asyncio
from fast_tts import AutoEngine

# Assume checkpoints/YourModelDir corresponds to a Spark, Orpheus, or Mega model
engine = AutoEngine(
    model_path="checkpoints/YourModelDir",
    snac_path=None,  # Required for Orpheus if `snac` subdirectory is not present
    lang="mandarin",  # Only used by Orpheus
    llm_device="cuda",  # LLM computation device
    tokenizer_device="cuda",  # Audio tokenizer device (Spark/Mega)
    detokenizer_device="cuda",  # Audio detokenizer device (Spark/Orpheus)
    backend="vllm",  # LLM backend accelerator
    torch_dtype="float32",  # LLM weight precision
    batch_size=1,
    llm_batch_size=256,
    wait_timeout=0.01,
    seed=42
)


async def main():
    # Basic synthesis example (Spark shown, but Orpheus/Mega also supported)
    wav = await engine.speak_async(
        text="Hello, world!",
        name="female",  # Spark built-in role “female”
        pitch="moderate",  # Spark only
        speed="moderate",  # Spark only
        temperature=0.9,
        top_k=50,
        top_p=0.95
    )
    engine.write_audio(wav, "output.wav")


if __name__ == '__main__':
    asyncio.run(main())
```

---

## Engine Auto-Detection Logic

`AutoEngine` determines the appropriate engine based on subdirectories under `model_path`:

- **Spark**: Detected when `LLM`, `BiCodec`, and `wav2vec2-large-xlsr-53` directories are all present.
- **Mega**: Detected when `aligner_lm`, `diffusion_transformer`, `duration_lm`, `g2p`, and `wavvae` directories are all
  present.
- **Orpheus**: Detected when a `snac` directory is present, or `snac_path` is provided externally.
- Otherwise, raises `RuntimeError: No engine found`.

---

## Initialization Parameters

| Parameter                     | Type                                                    | Default   | Description                                                              |
|-------------------------------|---------------------------------------------------------|-----------|--------------------------------------------------------------------------|
| `model_path`                  | `str`                                                   | —         | Path to the model root directory, containing engine-specific subfolders. |
| `max_length`                  | `int`                                                   | `32768`   | Max context length for LLM. Default `8192` for Orpheus.                  |
| `snac_path`                   | `Optional[str]`                                         | `None`    | Path to Orpheus SNAC weights; can be omitted if `snac` folder exists.    |
| `lang`                        | `Literal["mandarin", "french", …, "english", None]`     | `None`    | Language option for Orpheus.                                             |
| `llm_device`                  | `Literal["cpu","cuda","mps","auto"] \| str`             | `"auto"`  | Computation device for the LLM module.                                   |
| `tokenizer_device`            | Same as above                                           | `"auto"`  | Tokenizer device (Spark/Mega).                                           |
| `detokenizer_device`          | Same as above                                           | `"auto"`  | Detokenizer device (Spark/Orpheus).                                      |
| `backend`                     | `Literal["vllm","llama-cpp","sglang","torch","mlx-lm"]` | `"torch"` | LLM backend accelerator. Supported by all engines.                       |
| `wav2vec_attn_implementation` | `Optional[Literal["sdpa","flash_attention_2","eager"]]` | `None`    | Wav2vec attention implementation (Spark).                                |
| `llm_attn_implementation`     | Same as above                                           | `None`    | Attention implementation for LLM.                                        |
| `torch_dtype`                 | `Literal['float16','bfloat16','float32','auto']`        | `"auto"`  | Weight precision; Mega supports any precision.                           |
| `llm_gpu_memory_utilization`  | `Optional[float]`                                       | `0.6`     | Max GPU memory usage for vllm/sglang.                                    |
| `cache_implementation`        | `Optional[str]`                                         | `None`    | Caching strategy name (used only when `backend="torch"`).                |
| `batch_size`                  | `int`                                                   | `1`       | Tokenizer/detokenizer batch size.                                        |
| `llm_batch_size`              | `int`                                                   | `256`     | LLM decoding batch size.                                                 |
| `wait_timeout`                | `float`                                                 | `0.01`    | Async wait timeout for tokenizer/detokenizer.                            |
| `seed`                        | `int`                                                   | `0`       | Random seed.                                                             |
| `**kwargs`                    | —                                                       | —         | Additional backend-specific parameters.                                  |

Initialization outputs a message like:

```
Initializing `AutoEngine(engine=spark)` with config: (model_path='…', max_length=32768, …)
```

and sets `self._engine` to an instance of `AsyncSparkEngine`, `AsyncOrpheusEngine`, or `AsyncMega3Engine`.

---

## Common Methods

These interfaces are supported by all engines:

### `write_audio(audio: np.ndarray, filepath: str)`

Writes `audio` to the specified file in WAV format.

### `list_roles() → list[str]`

Returns a list of available speaker roles:

- Spark: List of added roles
- Orpheus: Built-in roles
- Mega: List of added roles

### `add_speaker(name: str, audio, reference_text: Optional[str]=None) → Awaitable`

Adds a custom speaker role for Spark/Orpheus/Mega:

- **Spark**: `audio` can be `bytes` or a file path. `reference_text` is optional.
- **Orpheus**: Not supported.
- **Mega**: `audio` must be a `(wav_path, npy_path)` tuple.

### `delete_speaker(name: str) → Awaitable`

Deletes a speaker by name.

---

## Core Synthesis and Cloning Interfaces

| Method                                                      | Description                                                   |
|-------------------------------------------------------------|---------------------------------------------------------------|
| `speak_async(...) → np.ndarray`                             | Async text-to-speech, returns full waveform.                  |
| `speak_stream_async(...) → AsyncIterator[np.ndarray]`       | Async streaming TTS, returns audio in chunks (Spark/Orpheus). |
| `clone_voice_async(...) → np.ndarray`                       | Async voice cloning from reference audio (Spark/Mega).        |
| `clone_voice_stream_async(...) → AsyncIterator[np.ndarray]` | Async streaming voice cloning (Spark only).                   |
| `multi_speak_async(...) → np.ndarray`                       | Multi-speaker dialogue synthesis using `<role:Name>`.         |
| `multi_speak_stream_async(...) → AsyncIterator[np.ndarray]` | Streaming version of multi-speaker synthesis.                 |

> **Parameter Notes (example: `speak_async`)**
> - `text: str`: Input text
> - `name: Optional[str]`: Speaker role name (Spark default: `"female"`)
> - `pitch, speed: Literal["very_low","low","moderate","high","very_high"]`: Spark only
> - `temperature: float = 0.9`: Sampling temperature
> - `top_k: int = 50`; `top_p: float = 0.95`; `repetition_penalty: float = 1.0`: Sampling parameters
> - `max_tokens: int = 4096`: Max token count for LLM generation
> - `length_threshold: int = 50`; `window_size: int = 50`: Long text splitting and window size
> - `split_fn: Optional[Callable[[str], list[str]]]`: Custom text splitting function

Other methods follow the same parameter structure. Refer to individual engine docs for details.

---

## Notes

1. **Engine Differences**:
    - **Spark** supports acoustic token reuse (`return_acoustic_tokens`, `acoustic_tokens`) and full streaming support.
    - **Orpheus** does not support voice cloning; focuses on long-form synthesis with emotional tags.
    - **Mega** requires WaveVAE features for cloning; supports multi-speaker synthesis.

2. **Performance Tuning**:
   Adjust `llm_device`, `backend`, `torch_dtype`, and memory usage parameters based on your hardware to improve speed
   and quality.

---

With `AutoEngine`, you can seamlessly switch between different TTS engines, simplifying model deployment and integration
significantly.