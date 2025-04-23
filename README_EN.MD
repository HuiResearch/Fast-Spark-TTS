# FastTTS Speech Synthesis and Cloning Platform 🔊

[中文](README.MD) | [English](README_EN.MD)

> 🚀 **FastTTS** – Based on models such as SparkTTS, OrpheusTTS and MegaTTS3, this platform provides high-quality Chinese
> speech
> synthesis and voice cloning services. With a simple and user-friendly web interface, you can effortlessly create
> natural
> and realistic human voices to suit a wide range of scenarios.

> If you find the project helpful, please consider giving it a star.

## ✨ Features

- 🚀 **Multiple Backend Acceleration**: Supports various acceleration strategies, including `vllm`, `sglang`,
  `llama cpp`, and `mlx-lm`.
- 🎯 **High Concurrency**: Uses dynamic batching to greatly improve concurrent processing.
- 🎛️ **Full Parameter Control**: Offers comprehensive adjustments for pitch, speed, timbre temperature, and more.
- 📱 **Lightweight Deployment**: Minimal dependencies with fast startup built on Flask and FastAPI.
- 🎨 **Clean Interface**: Features a modern, standardized UI.
- 🔊 **Long Text Speech Synthesis**: Capable of synthesizing long texts while maintaining consistent timbre.
- 🔄 **Streaming Speech Synthesis Support**: Provides real-time speech synthesis that plays as it generates, reducing
  wait times and enhancing interactive experiences.
- 🎭 **Multi-Character Speech Synthesis**: Supports synthesis for multiple characters, which can be used for scripted
  dialogues and more.

## 🖼️ Usage Example

[View the usage example demo](https://github.com/user-attachments/assets/ab7ca580-45b3-41ba-acfd-b2f68ff62948)

## 🛠️ Quick Start

### System Requirements

- Python 3.10+
- FastAPI
- vllm **or** sglang **or** llama-cpp

### Download Weights

|            模型             |                                                                                                             huggingface                                                                                                              |                                        modelscope                                         |                                       gguf                                       |
|:-------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
|         Spark-TTS         |                                                                            [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)                                                                             |    [SparkAudio/Spark-TTS-0.5B](https://modelscope.cn/models/SparkAudio/Spark-TTS-0.5B)    |    [SparkTTS-LLM-GGUF](https://huggingface.co/mradermacher/SparkTTS-LLM-GGUF)    | 
|        Orpheus-TTS        |                                  [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) & [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz)                                   | [canopylabs/orpheus-3b-0.1-ft](https://modelscope.cn/models/canopylabs/orpheus-3b-0.1-ft) | [orpheus-gguf](https://huggingface.co/isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF) | 
| Orpheus-TTS(Multilingual) | [orpheus-multilingual-research-release](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba)  & [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz) |                                             -                                             |                                        -                                         |
|         MegaTTS3          |                                                                                   [ByteDance/MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3)                                                                                    |                                             -                                             |                                        -                                         |

### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Inference Engine Installation

*Install one of the following as needed. If you use Torch for inference, you can skip these additional setups.*

- **vLLM**  
  (Ensure the vLLM version is greater than `0.7.2`.)
  ```bash
  pip install vllm
  ```  
  For details, refer to: https://github.com/vllm-project/vllm

- **llama-cpp**
  ```bash
  pip install llama-cpp-python
  ```  
  You can download the pre-converted gguf weights from the [Download Weights](#download-weights) section. Place the
  Spark-TTS weights in the subdirectory `LLM` and the Orpheus-TTS weights in the model directory. If you want to convert
  them yourself, use the `convert_hf_to_gguf.py` script as follows:

  ```bash
  git clone https://github.com/ggml-org/llama.cpp.git
  
  cd llama.cpp
  
  python convert_hf_to_gguf.py Spark-TTS-0.5B/LLM --outfile Spark-TTS-0.5B/LLM/model.gguf
  ```

- **sglang**
  ```bash
  pip install sglang
  ```  
  For more details, refer to: https://github.com/sgl-project/sglang

- **mlx-lm**  
  For running large language models on Apple Silicon (suitable for macOS):
  ```bash
  pip install mlx-lm
  ```  
  For more details, refer to: https://github.com/ml-explore/mlx-lm

### Starting the Service

1. **Clone the Repository**

   ```bash
   git clone https://github.com/HuiResearch/Fast-Spark-TTS.git
   cd Fast-Spark-TTS
   ```

2. **Start the SparkTTS API Service**

   The `backend` option can be selected based on your environment. Currently supported backends include `torch`, `vllm`,
   `sglang`, `llama-cpp`, and `mlx-lm`.

   ```bash
   python server.py \
   --model_path Spark-TTS-0.5B \  # Modify this to your model directory as needed
   --backend vllm \  # Choose one: vllm, sglang, torch, llama-cpp, or mlx-lm
   --llm_device cuda \
   --tokenizer_device cuda \
   --detokenizer_device cuda \
   --wav2vec_attn_implementation sdpa \
   --llm_attn_implementation sdpa \  # Recommended for torch engine acceleration
   --torch_dtype "bfloat16" \   # For spark-tts models, devices that do not support bfloat16 can only be set to float32.
   --max_length 32768 \
   --llm_gpu_memory_utilization 0.6 \
   --host 0.0.0.0 \
   --port 8000
   ```

   Access the page in your browser

    ```
    http://localhost:8000
    ```

   View the API documentation

    ```
    http://localhost:8000/docs
    ```

## 🚀 User Guide

### Speech Synthesis

1. Switch to the "Speech Synthesis" tab.
2. Enter the text you wish to convert into speech.
3. Adjust parameters such as gender, pitch, and speed.
4. Click the "Generate Speech" button.
5. Once processing completes, you can play or download the generated audio.

### Voice Cloning

1. Switch to the "Voice Cloning" tab.
2. Enter the target text.
3. Upload a reference audio file.
4. Provide the corresponding text for the reference audio.
5. Adjust the parameters.
6. Click the "Clone Voice" button.
7. Once cloning is complete, you can play or download the audio.

### Character Cloning

1. Switch to the "Character Cloning" tab.
2. Enter the target text.
3. Choose your preferred character.
4. Adjust the parameters.
5. Click the "Character Cloning" button.
6. After processing, you can play or download the generated audio.

## Inference Speed

- **GPU:** `A800`
- **Model:** `Spark-TTS-0.5B`
- Testing parameters and evaluation methods can be found in [speed_test.py](speed_test.py).
- Note: Both the output audio length (in seconds) and the inference time (in seconds) are measured in seconds.
- Evaluations include both long text and short text scenarios.
- The official code has not been benchmarked; please perform your own evaluations if necessary.

|  Scenario  |  engine   | device | audio len | cost time |  RTF  |
|:----------:|:---------:|:------:|:---------:|:---------:|:-----:|
| Short Text | llama-cpp |  cpu   |   7.48    |   6.808   | 0.910 |
| Short Text |   torch   |  gpu   |   7.18    |   7.675   | 1.069 |
| Short Text |   vllm    |  gpu   |   7.24    |   1.664   | 0.230 |
| Short Text |  sglang   |  gpu   |   7.58    |   1.073   | 0.142 |
| Long Text  | llama-cpp |  cpu   |  121.98   |  117.828  | 0.966 |
| Long Text  |   torch   |  gpu   |   113.7   |  107.167  | 0.943 |
| Long Text  |   vllm    |  gpu   |  111.82   |   7.282   | 0.065 |
| Long Text  |  sglang   |  gpu   |  117.02   |   4.197   | 0.036 |

## Usage Tips

- The `SparkTTS` model cannot be used if its weights are quantized to `float16`; please set `torch_dtype` to `bfloat16`
  or `float32`.
- If you experience extended periods of silence, try adjusting the `repetition_penalty` parameter. Values greater than
  `1.0` reduce the probability of generating repeated tokens, which helps to avoid long durations of silent token
  generation (the silent token id for SparkTTS is `163406`).
- `OrpheusTTS` supports emotion tags. You can simply add `<tag>` directly in the text to activate this feature. For the
  list of supported tags for each model, please refer to LANG_MAP
  in [orpheus_engine.py](fast_tts/engine/orpheus_engine.py).
- For security reasons, the **MegaTTS3** team has not uploaded the WaveVAE encoder parameters. Therefore, reference
  audio can only be downloaded from the following link for
  inference: [Reference Audio](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr).  
  If you wish to use your own audio, please refer to the instructions in the MegaTTS3
  project: [MegaTTS3](https://github.com/bytedance/MegaTTS3/tree/main?tab=readme-ov-file#inference).

## Reference

1. [Spark-TTS](https://github.com/SparkAudio/Spark-TTS)
2. [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
3. [MegaTTS3](https://github.com/bytedance/MegaTTS3)

## ⚠️ Disclaimer

This project provides a zero-shot voice cloning TTS model intended for academic research, educational purposes, and
legitimate applications such as personalized speech synthesis, assistive technologies, and language research.

Please note:

- Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deep fakes, or any other illegal
  activities.
- Ensure that you comply with local laws and ethical standards when using this model.
- The developers are not responsible for any misuse of this model.

This project advocates responsible AI development and use, and encourages the community to uphold safety and ethical
principles in AI research and applications.

## License and Acknowledgements

This project is built upon [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) and is distributed under the same
open-source license as SparkTTS. For details, please refer to the
original [SparkTTS License](https://github.com/SparkAudio/Spark-TTS/blob/main/LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HuiResearch/Fast-Spark-TTS&type=Date)](https://www.star-history.com/#HuiResearch/Fast-Spark-TTS&Date)
