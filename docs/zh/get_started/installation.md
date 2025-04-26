## Flash-TTS 安装文档

> 本文档详细介绍 Flash-TTS 推理引擎的安装与部署流程，覆盖环境要求、模型权重下载、依赖安装。

---

### 环境要求

- **Python**: 3.10+
- **操作系统**: Linux x86_64、macOS 或 Windows（推荐 WSL2）
- **必需依赖**:
    - `fastapi`
    - 推理后端选项之一: `vllm`、`sglang`、`llama-cpp-python`、`mlx-lm`

---

### 下载权重

|            模型             |                                                                                                             huggingface                                                                                                              |                                        modelscope                                         |                                       gguf                                       |
|:-------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
|         Spark-TTS         |                                                                            [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)                                                                             |    [SparkAudio/Spark-TTS-0.5B](https://modelscope.cn/models/SparkAudio/Spark-TTS-0.5B)    |    [SparkTTS-LLM-GGUF](https://huggingface.co/mradermacher/SparkTTS-LLM-GGUF)    | 
|        Orpheus-TTS        |                                  [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) & [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz)                                   | [canopylabs/orpheus-3b-0.1-ft](https://modelscope.cn/models/canopylabs/orpheus-3b-0.1-ft) | [orpheus-gguf](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF) | 
| Orpheus-TTS(Multilingual) | [orpheus-multilingual-research-release](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba)  & [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz) |                                             -                                             |                                        -                                         |
|         MegaTTS3          |                                                                                   [ByteDance/MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3)                                                                                    |                                             -                                             |                                        -                                         |

---

### 安装依赖

#### 1. 安装`torch`和`torchaudio`

进入[pytorch官网](https://pytorch.org/get-started/locally/)，查找对应的设备环境安装命令，执行。请注意自己的设备的`cuda驱动`
版本等信息。

以`cuda 12.4`为例：

```bash
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

#### 2. 安装`flashtts`

- **pip**安装方式：
  ```bash
  pip install flashtts
  ```
  
- **源码安装**：
  ```bash
  git clone https://github.com/HuiResearch/FlashTTS.git
  cd FlashTTS
  pip install .
  ```

如果windows环境遇到安装`WeTextProcessing`出错，需要vs c++编译器时，可以使用`conda`先安装`pynini==2.1.6`:

```bash
conda install -c conda-forge pynini==2.1.6
pip install WeTextProcessing==1.0.4.1
```

#### 3. 推理后端安装（按需选择一项）

- **vLLM** (版本需 > 0.7.2)

  `cuda 12.4`默认安装命令为：
  ```bash
  pip install vllm
  ```
  如果`cuda`为其它版本，请参考: https://docs.vllm.ai/en/latest/getting_started/installation.html


- **llama-cpp-python**
  ```bash
  pip install llama-cpp-python
  ```
    - 若使用 GGUF 权重，请将 `model.gguf` 放至 `checkpoints/<model>/LLM/` 目录。
    - 如需自行转换，可参考:
      ```bash
      git clone https://github.com/ggml-org/llama.cpp.git
      cd llama.cpp
      python convert_hf_to_gguf.py Spark-TTS-0.5B/LLM --outfile Spark-TTS-0.5B/LLM/model.gguf
      ```

- **sglang**
  ```bash
  pip install sglang
  ```
  参考: https://docs.sglang.ai/start/install.html

- **mlx-lm** (适用于 Apple Silicon macOS)
  ```bash
  pip install mlx-lm
  ```
  参考: https://github.com/ml-explore/mlx-lm

