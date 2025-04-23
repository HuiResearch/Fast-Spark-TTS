# FastTTS 语音合成与克隆平台 🔊

[中文](README.MD) | [English](README_EN.MD)

> 🚀 **FastTTS** - 基于SparkTTS、OrpheusTTS、MegaTTS3等模型，提供高质量中文语音合成与声音克隆服务。通过简单易用的Web界面，让您轻松创建自然逼真的人声，满足各种场景需求。

> 如果项目对您有帮助，欢迎给项目点个star。

## ✨ 特性

- 🚀 **多种后端加速**: 支持`vllm`、`sglang`、`llama cpp`、`mlx-lm` 多种加速策略
- 🎯 **高并发**: 采用动态批处理，极大提高并发
- 🎛️ **全参数控制**: 音调、语速、音色温度等全方位可调
- 📱 **轻量部署**: 最小依赖，基于Flask和fastapi快速启动
- 🎨 **简洁界面**: 标准的现代化UI
- 🔊 **长文本语音合成**: 实现超长文本语音合成，并保持音色一致性。
- 🔄 **支持流式语音合成**: 实现实时语音合成，边生成边播放，降低等待时间，提升交互体验
- 🎭 **多角色语音合成**: 支持多角色语音合成，可应用于剧本对话合成。

## 🖼️ 使用示例

https://github.com/user-attachments/assets/ab7ca580-45b3-41ba-acfd-b2f68ff62948

## 🛠️ 快速开始

### 环境要求

- Python 3.10+
- fastapi
- vllm 或 sglang 或 llama-cpp

### 下载权重

|            模型             |                                                                                                             huggingface                                                                                                              |                                        modelscope                                         |                                       gguf                                       |
|:-------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
|         Spark-TTS         |                                                                            [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)                                                                             |    [SparkAudio/Spark-TTS-0.5B](https://modelscope.cn/models/SparkAudio/Spark-TTS-0.5B)    |    [SparkTTS-LLM-GGUF](https://huggingface.co/mradermacher/SparkTTS-LLM-GGUF)    | 
|        Orpheus-TTS        |                                  [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) & [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz)                                   | [canopylabs/orpheus-3b-0.1-ft](https://modelscope.cn/models/canopylabs/orpheus-3b-0.1-ft) | [orpheus-gguf](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF) | 
| Orpheus-TTS(Multilingual) | [orpheus-multilingual-research-release](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba)  & [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz) |                                             -                                             |                                        -                                         |
|         MegaTTS3          |                                                                                   [ByteDance/MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3)                                                                                    |                                             -                                             |                                        -                                         |

### 安装依赖

```bash
pip install -r requirements.txt
```

推理引擎安装 (按需求安装一个即可，若是使用torch推理，则可以跳过)

- **vLLM**

  vllm版本需要大于`0.7.2`
    ```bash
    pip install vllm
    ```
  具体参考链接：https://github.com/vllm-project/vllm


- **llama-cpp**
    ```bash
    pip install llama-cpp-python
    ```

  可以从[下载权重](#下载权重)下载已经转好的gguf权重。spark-tts请将下载后的权重放至模型子目录`LLM`
  下，orpheus-tts放至模型目录下。如果想要自行转换，请使用`convert_hf_to_gguf.py`脚本转换权重，操作方式如下：

    ```bash
    git clone https://github.com/ggml-org/llama.cpp.git
    
    cd llama.cpp
    
    python convert_hf_to_gguf.py Spark-TTS-0.5B/LLM --outfile Spark-TTS-0.5B/LLM/model.gguf
    ```

- **sglang**

    ```bash
    pip install sglang
    ```

  具体参考链接：https://github.com/sgl-project/sglang


- **mlx-lm**

  在Apple Silicon上运行大型语言模型，适用于MacOS。
    ```bash
    pip install mlx-lm
    ```
  具体参考链接：https://github.com/ml-explore/mlx-lm

### 启动

1. 克隆项目仓库

```bash
git clone https://github.com/HuiResearch/Fast-Spark-TTS.git
cd Fast-Spark-TTS
```

2. 启动SparkTTS API服务

`backend`可根据自己的环境，选择对应的推理引擎。目前支持`torch`、`vllm`、`sglang`、`llama-cpp`、`mlx-lm`。

```bash
python server.py \
--model_path Spark-TTS-0.5B \  # 可修改为自己的模型地址
--backend vllm \  # vllm、sglang、torch、llama-cpp、mlx-lm任选一个
--llm_device cuda \
--tokenizer_device cuda \
--detokenizer_device cuda \
--wav2vec_attn_implementation sdpa \
--llm_attn_implementation sdpa \  # 如果使用torch engine，最好开启加速
--torch_dtype "bfloat16" \   # 对于spark-tts模型，不支持bfloat16的设备，只能设置为float32.
--max_length 32768 \
--llm_gpu_memory_utilization 0.6 \
--host 0.0.0.0 \
--port 8000
```

在浏览器中访问页面

```
http://localhost:8000
```

查看api文档

```
http://localhost:8000/docs
```

## 🚀 使用指南

### 语音合成

1. 切换到「语音合成」标签页
2. 输入需要转换为语音的文本
3. 调整性别、音调、语速等参数
4. 点击「生成语音」按钮
5. 等待生成完成后即可播放或下载

### 声音克隆

1. 切换到「声音克隆」标签页
2. 输入目标文本
3. 上传参考音频
4. 输入参考音频对应的文本
5. 调整参数
6. 点击「克隆声音」按钮
7. 等待克隆完成后即可播放或下载

### 角色克隆

1. 切换到「角色克隆」标签页
2. 输入目标文本
3. 选择自己喜欢的角色
4. 调整参数
5. 点击「角色克隆」按钮
6. 等待克隆完成后即可播放或下载

## 推理速度

- 显卡：`A800`
- 模型：`Spark-TTS-0.5B`
- 测试使用参数和评测方式可参考[speed_test.py](speed_test.py)
- 输出音频长度(audio len)、推理耗时(cost time)单位都为秒。
- 评测包括长文本和短文本场景。
- 官方代码未评测，如有需求，可自行评测。

| 场景  |  engine   | device | audio len | cost time |  RTF  |
|:---:|:---------:|:------:|:---------:|:---------:|:-----:|
| 短文本 | llama-cpp |  cpu   |   7.48    |   6.808   | 0.910 |
| 短文本 |   torch   |  gpu   |   7.18    |   7.675   | 1.069 |
| 短文本 |   vllm    |  gpu   |   7.24    |   1.664   | 0.230 |
| 短文本 |  sglang   |  gpu   |   7.58    |   1.073   | 0.142 |
| 长文本 | llama-cpp |  cpu   |  121.98   |  117.828  | 0.966 |
| 长文本 |   torch   |  gpu   |   113.7   |  107.167  | 0.943 |
| 长文本 |   vllm    |  gpu   |  111.82   |   7.282   | 0.065 |
| 长文本 |  sglang   |  gpu   |  117.02   |   4.197   | 0.036 |

## 使用提示

- `SparkTTS`模型权重量化为`float16`无法使用，`torch_dtype`请设置为`bfloat16`或`float32`。
- 如果出现长空白音，可尝试调整`repetition_penalty`参数，大于`1.0`会降低重复token生成的概率，可避免长时间静音`token`生成(
  SparkTTS的静音token id为`163406`)。
- `OrpheusTTS`支持emotion tag，直接在文本中添加`<tag>`
  即可实现。每个模型支持的tag标签请参考：[orpheus_engine.py](fast_tts/engine/orpheus_engine.py) 中 LANG_MAP
- 出于安全考虑，`MegaTTS3`
  团队未上传WaveVAE编码器参数。因此，只能从此处链接下载参考音频进行推理：[参考音频](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr)
  。如果需要使用自己的音频，请参考MegaTTS3项目的说明：[MegaTTS3](https://github.com/bytedance/MegaTTS3/tree/main?tab=readme-ov-file#inference)

## Reference

1. [Spark-TTS](https://github.com/SparkAudio/Spark-TTS)
2. [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
3. [MegaTTS3](https://github.com/bytedance/MegaTTS3)

## ⚠️ 免责声明

该项目提供了零样本语音克隆 TTS 模型，旨在用于学术研究、教育目的和合法应用，例如个性化语音合成、辅助技术和语言研究。

请注意：

- 请勿将此模型用于未经授权的语音克隆、冒充、欺诈、诈骗、深度伪造或任何非法活动。

- 确保使用此模型时遵守当地的法律法规并遵守道德标准。

- 开发人员对于此模型的任何误用不承担任何责任。

本项目提倡负责任地开发和使用人工智能，并鼓励社区在人工智能研究和应用中坚持安全和道德原则。

## 许可协议与致谢

本项目基于 [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) 构建，并采用与 SparkTTS
相同的开源许可协议进行分发。详情请参阅原始 [SparkTTS 许可协议](https://github.com/SparkAudio/Spark-TTS/blob/main/LICENSE)。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HuiResearch/Fast-Spark-TTS&type=Date)](https://www.star-history.com/#HuiResearch/Fast-Spark-TTS&Date)