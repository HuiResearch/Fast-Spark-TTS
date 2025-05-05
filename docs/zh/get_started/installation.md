## Flash-TTS 安装指南

> 本文档详细介绍 Flash-TTS 推理引擎的安装与部署流程，包括环境要求、模型权重下载及依赖安装步骤。

---

### 环境要求

* **Python**：3.10 及以上版本
* **操作系统**：Linux x86\_64、macOS，或 Windows（推荐使用 WSL2）
* **必备依赖**：

    * `fastapi`
    * 至少一种推理后端：`vllm`、`sglang`、`llama-cpp-python`、`mlx-lm` 或 `tensorrt-llm`

---

### 模型权重下载

|            模型             |                                                                                                             huggingface                                                                                                              |                                        modelscope                                         |                                       gguf                                       |
|:-------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
|         Spark-TTS         |                                                                            [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)                                                                             |    [SparkAudio/Spark-TTS-0.5B](https://modelscope.cn/models/SparkAudio/Spark-TTS-0.5B)    |    [SparkTTS-LLM-GGUF](https://huggingface.co/mradermacher/SparkTTS-LLM-GGUF)    | 
|        Orpheus-TTS        |                                  [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) & [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz)                                   | [canopylabs/orpheus-3b-0.1-ft](https://modelscope.cn/models/canopylabs/orpheus-3b-0.1-ft) | [orpheus-gguf](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF) | 
| Orpheus-TTS(Multilingual) | [orpheus-multilingual-research-release](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba)  & [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz) |                                             -                                             |                                        -                                         |
|         MegaTTS3          |                                                                                   [ByteDance/MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3)                                                                                    |                                             -                                             |                                        -                                         |

---

### 安装依赖

#### 1. 安装 PyTorch

请前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适配您系统和 CUDA 版本的安装命令。
例如，针对 CUDA 12.4：

```bash
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

#### 2. 安装 Flash-TTS

* 使用 pip 安装：

```bash
pip install flashtts
```

* 或通过源码安装：

```bash
git clone https://github.com/HuiResearch/FlashTTS.git
cd FlashTTS
pip install .
```

> **Windows 用户提示**：
> 若在安装 `WeTextProcessing` 时遇到编译错误，可先通过 Conda 安装依赖：

```bash
conda install -c conda-forge pynini==2.1.6
pip install WeTextProcessing==1.0.4.1
```

---

### 安装推理后端（按需选择）

#### vLLM （推荐）

* 需版本 ≥ 0.7.2。针对 CUDA 12.4：

```bash
pip install vllm
```

* 其它版本请参考：[vLLM 官方文档](https://docs.vllm.ai/en/latest/getting_started/installation.html)

---

#### llama-cpp-python

```bash
pip install llama-cpp-python
```

* 如果使用 GGUF 格式权重，请将 `model.gguf` 文件放入 `checkpoints/<model>/LLM/` 路径下。
* 若需转换权重，可参考以下命令：

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

* 更多信息参考：[sglang 安装指南](https://docs.sglang.ai/start/install.html)

---

#### mlx-lm（仅支持 Apple Silicon）

```bash
pip install mlx-lm
```

* 更多信息参考：[mlx-lm 项目主页](https://github.com/ml-explore/mlx-lm)

---

#### TensorRT-LLM

以 CUDA 12.4 为例：

```bash
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu124
```

> 注意事项：
>
> * 当前 TensorRT-LLM Windows 仅支持 Python 3.10。
> * Windows环境最新支持版本：0.16.0（截止 2025-05-05）
> * 详情见：[NVIDIA PyPI 仓库](https://pypi.nvidia.com)

安装验证：

```bash
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

##### 转换 LLM 权重为 TensorRT Engine

1. 参考官方模型转换文档：
   [https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/)

2. 选择与所用模型匹配的类型（例如：Spark-TTS 使用 `qwen`）。

3. 转换完成后，将输出的 Engine 文件夹重命名为 `tensorrt-engine`，并移动至模型路径中，例如：

```bash
Spark-TTS-0.5B/LLM/tensorrt-engine
```

此后即可在 Flash-TTS 中加载该模型进行推理。

当然可以，以下是优化后文档结尾附加的**环境支持表**，清晰展示了不同操作系统下可用的后端：

---

### 环境支持矩阵

| 推理后端           | Linux ✅ | Windows ✅ | macOS ✅ | 说明                                  |
|----------------|:-------:|:---------:|:-------:|-------------------------------------|
| `vllm`         |    ✅    |     ❌     |    ❌    | 仅支持 Linux，需 CUDA 环境                 |
| `sglang`       |    ✅    |     ❌     |    ❌    | 仅支持 Linux，适配大多数 GPU                 |
| `tensorrt-llm` |    ✅    |    ⚠️     |    ❌    | Windows 仅支持 Python 3.10，版本 ≤ 0.16.0 |
| `llama-cpp`    |    ✅    |     ✅     |    ✅    | 支持 GGUF 权重格式，跨平台                    |
| `mlx-lm`       |    ❌    |     ❌     |    ✅    | 仅支持 macOS（Apple Silicon）            |
| `torch`        |    ✅    |     ✅     |    ✅    | 核心依赖，所有平台均支持                        |

> ⚠️ **说明**：
>
> * Windows 下推荐使用 **WSL2** 获取完整 Linux 功能支持。
> * 若在 macOS 使用非 Apple Silicon 芯片，`mlx-lm` 将不可用。


