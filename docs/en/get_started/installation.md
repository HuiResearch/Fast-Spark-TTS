## Fast-TTS Installation Guide

> This document provides a detailed walkthrough for installing and deploying the Fast-TTS inference engine, including
> environment requirements, model weight downloads, and dependency installation.

---

### Environment Requirements

- **Python**: 3.10+
- **Operating System**: Linux x86_64, macOS, or Windows (WSL2 recommended)
- **Required Dependencies**:
    - `fastapi`
    - One of the supported inference backends: `vllm`, `sglang`, `llama-cpp-python`, `mlx-lm`

---

### Downloading Model Weights

|           Model            |                                                                                                             HuggingFace                                                                                                             |                                        ModelScope                                         |                                       GGUF                                       |
|:--------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
|         Spark-TTS          |                                                                            [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)                                                                            |    [SparkAudio/Spark-TTS-0.5B](https://modelscope.cn/models/SparkAudio/Spark-TTS-0.5B)    |    [SparkTTS-LLM-GGUF](https://huggingface.co/mradermacher/SparkTTS-LLM-GGUF)    |
|        Orpheus-TTS         |                                  [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) & [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz)                                  | [canopylabs/orpheus-3b-0.1-ft](https://modelscope.cn/models/canopylabs/orpheus-3b-0.1-ft) | [orpheus-gguf](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF) |
| Orpheus-TTS (Multilingual) | [orpheus-multilingual-research-release](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba) & [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz) |                                             -                                             |                                        -                                         |
|          MegaTTS3          |                                                                                   [ByteDance/MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3)                                                                                   |                                             -                                             |                                        -                                         |

---

### Installing Dependencies

#### 1. Install `torch` and `torchaudio`

Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to find the correct install command for
your environment. Make sure to check your CUDA version and other device details.

For example, with CUDA 12.4:

```bash
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

#### 2. Install Core Dependencies

```bash
# Clone the repository and navigate to the directory
git clone https://github.com/HuiResearch/Fast-Spark-TTS.git
cd Fast-Spark-TTS

# Install core dependencies
pip install -r requirements.txt
```

If you encounter an error installing `WeTextProcessing` in a Windows environment due to the need for a VS C++ compiler, you can first install `pynini==2.1.6` using `conda`:

```bash
conda install -c conda-forge pynini==2.1.6
pip install WeTextProcessing==1.0.4.1
```

#### 3. Install Inference Backend (Choose One)

- **vLLM** (version > 0.7.2)

  Default installation command for CUDA 12.4:
  ```bash
  pip install vllm
  ```
  For other CUDA versions, refer to: https://docs.vllm.ai/en/latest/getting_started/installation.html

- **llama-cpp-python**
  ```bash
  pip install llama-cpp-python
  ```
    - If using GGUF weights, place `model.gguf` in the `checkpoints/<model>/LLM/` directory.
    - To convert manually:
      ```bash
      git clone https://github.com/ggml-org/llama.cpp.git
      cd llama.cpp
      python convert_hf_to_gguf.py Spark-TTS-0.5B/LLM --outfile Spark-TTS-0.5B/LLM/model.gguf
      ```

- **sglang**
  ```bash
  pip install sglang
  ```
  Reference: https://docs.sglang.ai/start/install.html

- **mlx-lm** (for Apple Silicon macOS)
  ```bash
  pip install mlx-lm
  ```
  Reference: https://github.com/ml-explore/mlx-lm