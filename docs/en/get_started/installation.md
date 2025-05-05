## Flash-TTS Installation Guide

> This document provides a detailed walkthrough for installing and deploying the Flash-TTS inference engine, including
> environment requirements, model weight downloads, and dependency installation steps.

---

### Environment Requirements

* **Python**: Version 3.10 or above
* **Operating System**: Linux x86\_64, macOS, or Windows (WSL2 is recommended)
* **Required Dependencies**:

    * `fastapi`
    * At least one inference backend: `vllm`, `sglang`, `llama-cpp-python`, `mlx-lm`, or `tensorrt-llm`

---

### Model Weight Downloads

|           Model            |                                                                                                             Hugging Face                                                                                                             |                                        ModelScope                                         |                                       GGUF                                       |
|:--------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
|         Spark-TTS          |                                                                            [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)                                                                             |    [SparkAudio/Spark-TTS-0.5B](https://modelscope.cn/models/SparkAudio/Spark-TTS-0.5B)    |    [SparkTTS-LLM-GGUF](https://huggingface.co/mradermacher/SparkTTS-LLM-GGUF)    |
|        Orpheus-TTS         |                                  [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) & [hubertsiuzdak/snac\_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz)                                  | [canopylabs/orpheus-3b-0.1-ft](https://modelscope.cn/models/canopylabs/orpheus-3b-0.1-ft) | [orpheus-gguf](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF) |
| Orpheus-TTS (Multilingual) | [orpheus-multilingual-research-release](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba) & [hubertsiuzdak/snac\_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz) |                                             -                                             |                                        -                                         |
|          MegaTTS3          |                                                                                   [ByteDance/MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3)                                                                                    |                                             -                                             |                                        -                                         |

---

### Dependency Installation

#### 1. Install PyTorch

Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the installation command suitable
for your system and CUDA version.
For example, for CUDA 12.4:

```bash
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

#### 2. Install Flash-TTS

* Install via pip:

```bash
pip install flashtts
```

* Or install from source:

```bash
git clone https://github.com/HuiResearch/FlashTTS.git
cd FlashTTS
pip install .
```

> **Windows User Notice**:
> If you encounter compilation errors when installing `WeTextProcessing`, you can install dependencies via Conda first:

```bash
conda install -c conda-forge pynini==2.1.6
pip install WeTextProcessing==1.0.4.1
```

---

### Install Inference Backends (choose as needed)

#### vLLM (Recommended)

* Version ≥ 0.7.2 is required. For CUDA 12.4:

```bash
pip install vllm
```

* For other versions, refer to
  the [vLLM official documentation](https://docs.vllm.ai/en/latest/getting_started/installation.html)

---

#### llama-cpp-python

```bash
pip install llama-cpp-python
```

* If using GGUF format weights, place the `model.gguf` file under the `checkpoints/<model>/LLM/` directory.
* To convert weights, use the following commands:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
python convert_hf_to_gguf.py Spark-TTS-0.5B/LLM --outfile Spark-TTS-0.5B/LLM/model.gguf
```

---

#### sglang

```bash
pip install sglang
```

* For more information, refer to the [sglang installation guide](https://docs.sglang.ai/start/install.html)

---

#### mlx-lm (Apple Silicon Only)

```bash
pip install mlx-lm
```

* More info: [mlx-lm GitHub project](https://github.com/ml-explore/mlx-lm)

---

#### TensorRT-LLM

Example for CUDA 12.4:

```bash
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu124
```

> **Notes**:
>
> * TensorRT-LLM on Windows currently supports only Python 3.10.
> * Latest supported version for Windows: 0.16.0 (as of 2025-05-05)
> * Details: [NVIDIA PyPI Repository](https://pypi.nvidia.com)

Verify installation:

```bash
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

##### Convert LLM Weights to TensorRT Engine

1. Refer to the official model conversion docs:
   [https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/)

2. Choose the appropriate model type (e.g., Spark-TTS uses `qwen`).

3. After conversion, rename the output engine folder to `tensorrt-engine` and move it to the model directory, for
   example:

```bash
Spark-TTS-0.5B/LLM/tensorrt-engine
```

Flash-TTS can now load and infer from the converted model.

---

### Backend Support Matrix

| Inference Backend | Linux ✅ | Windows ✅ | macOS ✅ | Notes                                               |
|-------------------|:-------:|:---------:|:-------:|-----------------------------------------------------|
| `vllm`            |    ✅    |     ❌     |    ❌    | Linux-only, requires CUDA                           |
| `sglang`          |    ✅    |     ❌     |    ❌    | Linux-only, supports most GPUs                      |
| `tensorrt-llm`    |    ✅    |    ⚠️     |    ❌    | Windows supports Python 3.10 only, version ≤ 0.16.0 |
| `llama-cpp`       |    ✅    |     ✅     |    ✅    | GGUF format supported, cross-platform               |
| `mlx-lm`          |    ❌    |     ❌     |    ✅    | macOS only (Apple Silicon)                          |
| `torch`           |    ✅    |     ✅     |    ✅    | Core dependency, supported on all platforms         |

> ⚠️ **Notes**:
>
> * On Windows, **WSL2** is recommended for full Linux feature support.
> * On macOS, `mlx-lm` is not available for non-Apple Silicon chips.
