<div align="center">
  <img src="docs/zh/_img/icon.png" width="450"/>

[📘Documentation](docs/en/README.MD)

[中文](README.MD) | [English](README_EN.MD)

</div>

---

## Introduction

> Powered by models like SparkTTS, OrpheusTTS, and MegaTTS3, this project provides high-quality Mandarin speech
> synthesis and voice cloning services. With an easy-to-use web interface, you can effortlessly generate natural,
> human-like voices for various use cases.

> If you find this project helpful, feel free to give it a star ⭐.

## ✨ Features

- 🚀 **Multiple backend accelerations**: Supports `vllm`, `sglang`, `llama cpp`, and `mlx-lm` for flexible inference
  strategies.
- 🎯 **High concurrency**: Dynamic batching greatly improves parallel processing capability.
- 🎛️ **Full parameter control**: Fully adjustable pitch, speed, timbre temperature, and more.
- 📱 **Lightweight deployment**: Minimal dependencies, fast startup based on Flask and FastAPI.
- 🎨 **Modern UI**: Clean and modern web interface.
- 🔊 **Long-form TTS**: Supports synthesis of very long texts while maintaining consistent voice quality.
- 🔄 **Streaming synthesis**: Real-time synthesis with immediate playback, reducing wait time and enhancing
  interactivity.
- 🎭 **Multi-role voice synthesis**: Enables synthesis of dialogue involving multiple characters.

## 🖼️ Frontend Showcase

https://github.com/user-attachments/assets/1bd9d586-fac7-4016-b955-5a58d8fb9d7e

## Voice Demos

Here are sample voice demos cloned using the Spark-TTS model.

---

<table>
<tr>
<td align="center">

**Donald Trump en**
</td>
<td align="center">

**Donald Trump zh**
</td>
</tr>

<tr>
<td align="center">

[Donald Trump en](https://github.com/user-attachments/assets/2e60851e-333a-4dc8-adf3-89a8b6102641)

</td>
<td align="center">

[Donald Trump zh](https://github.com/user-attachments/assets/110d105b-83b9-4482-a43e-41838d13c7ba)

</td>
</tr>
</table>

<table>
<tr>
<td align="center">

**哪吒**
</td>
<td align="center">

**李靖**
</td>
</tr>

<tr>
<td align="center">

[哪吒](https://github.com/user-attachments/assets/44f6e9d9-77ba-4f0f-94e4-e7cac0ef925c)

</td>
<td align="center">

[李靖](https://github.com/user-attachments/assets/55a9be10-10cc-4438-a507-a6a8cabe47b8)

</td>
</tr>
</table>

<table>
<tr>
<td align="center">

**余承东**
</td>
<td align="center">

**徐志胜**
</td>
</tr>

<tr>
<td align="center">

[余承东](https://github.com/user-attachments/assets/f1b8d59c-6d28-405a-a91b-9b4514daf7ff)

</td>
<td align="center">

[徐志胜](https://github.com/user-attachments/assets/465eb52e-4692-4d17-aabb-6dd14664af3b)

</td>
</tr>
</table>

---

## Inference Speed

- GPU: `A800`
- Model: `Spark-TTS-0.5B`
- Refer to [speed_test.py](examples/speed_test.py) for test parameters and methods.
- Audio length and inference time are measured in seconds.
- Evaluated on both short and long text scenarios.
- Official codes not benchmarked; feel free to run your own tests if needed.

| Scenario |  Engine   | Device | Audio Length | Cost Time |  RTF  |
|:--------:|:---------:|:------:|:------------:|:---------:|:-----:|
|  Short   | llama-cpp |  CPU   |     7.48     |   6.808   | 0.910 |
|  Short   |   torch   |  GPU   |     7.18     |   7.675   | 1.069 |
|  Short   |   vllm    |  GPU   |     7.24     |   1.664   | 0.230 |
|  Short   |  sglang   |  GPU   |     7.58     |   1.073   | 0.142 |
|   Long   | llama-cpp |  CPU   |    121.98    |  117.828  | 0.966 |
|   Long   |   torch   |  GPU   |    113.7     |  107.167  | 0.943 |
|   Long   |   vllm    |  GPU   |    111.82    |   7.282   | 0.065 |
|   Long   |  sglang   |  GPU   |    117.02    |   4.197   | 0.036 |

## Notes

1. The `SparkTTS` model does **not** support weight quantization to `float16`. Use `bfloat16` or `float32` for
   `torch_dtype`.
2. If there's a long silent segment in output, try adjusting `repetition_penalty` > 1.0 to reduce the chance of
   repeating silent tokens (silent token ID for SparkTTS is `163406`).
3. `OrpheusTTS` supports emotion tags. Just insert `<tag>` directly into the text. Refer to supported tags in
   `[orpheus_engine.py](fast_tts/engine/orpheus_engine.py)` via `LANG_MAP`.
4. For security reasons, the `MegaTTS3` team hasn't uploaded WaveVAE encoder parameters. Use the provided reference
   audios here for
   inference: [Reference Audio](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr). For custom
   audio, follow the official
   instructions: [MegaTTS3](https://github.com/bytedance/MegaTTS3/tree/main?tab=readme-ov-file#inference)

## Acknowledgements

1. [Spark-TTS](https://github.com/SparkAudio/Spark-TTS)
2. [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
3. [MegaTTS3](https://github.com/bytedance/MegaTTS3)

## ⚠️ Disclaimer

This project offers a zero-shot voice cloning TTS model intended for academic research, education, and legitimate
applications such as personalized voice generation, assistive tech, and linguistic studies.

Please note:

- Do **not** use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal
  activities.
- Always comply with local laws and ethical guidelines.
- The developers are **not** responsible for any misuse of this model.

We advocate responsible AI development and encourage the community to uphold safety and ethical standards in AI research
and deployment.

## License & Attribution

This project is built on [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) and follows the same open-source
license.  
See [SparkTTS License](https://github.com/SparkAudio/Spark-TTS/blob/main/LICENSE) for details.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HuiResearch/Fast-Spark-TTS&type=Date)](https://www.star-history.com/#HuiResearch/Fast-Spark-TTS&Date)
