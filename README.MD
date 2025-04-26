<div align="center">
  <img src="docs/zh/_img/icon.png" width="450" alt="FlashTTS Logo"/>

[📘 Documentation](docs/zh/README.MD)

[中文](README.MD) | [English](README_EN.MD)

</div>

---

# FlashTTS

> 基于 **SparkTTS、OrpheusTTS、MegaTTS&nbsp;3** 等先进模型，FlashTTS 提供高质量的中文语音合成与零样本声音克隆服务。借助简洁直观的
> Web 界面，您可以快速生成自然逼真的人声，满足配音、阅读、无障碍、虚拟角色等多元场景需求。

> 如果 FlashTTS 对你有帮助，欢迎留下一个 ⭐ Star！

## ✨ 功能亮点

|     | 功能          | 说明                                                 |
|-----|-------------|----------------------------------------------------|
| 🚀  | **多后端推理加速** | 支持 `vllm`、`sglang`、`llama‑cpp`、`mlx‑lm` 等多种高性能推理引擎 |
| 🎯  | **高并发**     | 动态批处理与异步队列，轻松应对大流量请求                               |
| 🎛️ | **全参数控制**   | 可调节音调、语速、温度、情感标签等                                  |
| 📱  | **轻量部署**    | 基于 FastAPI，一条命令即可启动；最小依赖                           |
| 🔊  | **长文本合成**   | 支持超长文本，保持连续音色一致                                    |
| 🔄  | **流式 TTS**  | 边生成边播放，降低等待，提高交互体验                                 |
| 🎭  | **多角色对话**   | 同文本多角色合成，适合剧本配音                                    |
| 🎨  | **现代化前端**   | 适配Web端                                             |

## 🖼️ 前端示例

https://github.com/user-attachments/assets/1bd9d586-fac7-4016-b955-5a58d8fb9d7e

## 🔈 语音演示

下列 Demo 展示了 FlashTTS 在不同模型与角色上的克隆效果。

### SparkTTS 模型

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

[Donald Trump en](https://github.com/user-attachments/assets/79d4d1d1-5579-4ddb-8092-500b3db05d45)

</td>
<td align="center">

[Donald Trump zh](https://github.com/user-attachments/assets/262f4306-4a0f-4735-b4e7-708cd3193324)

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

[哪吒](https://github.com/user-attachments/assets/4743778b-63c7-4c42-95b2-636828364d67)

</td>
<td align="center">

[李靖](https://github.com/user-attachments/assets/b5a51fa9-c5a1-4f39-bbf8-09e73b2c1936)

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

[余承东](https://github.com/user-attachments/assets/1cd5f156-010d-4bc9-8e99-f37def54b99f)

</td>
<td align="center">

[徐志胜](https://github.com/user-attachments/assets/f197ac39-5928-4ead-83d7-071b1bf5f974)

</td>
</tr>
</table>

### MegaTTS&nbsp;3 模型

<table>
<tr>
<td align="center">

**蔡徐坤**
</td>
<td align="center">

**太乙真人**
</td>
</tr>

<tr>
<td align="center">

[蔡徐坤](https://github.com/user-attachments/assets/bcb48ead-f157-45d7-8060-cbccf418f0ed)

</td>
<td align="center">

[太乙真人](https://github.com/user-attachments/assets/73edbd6b-ece3-4e22-95a2-53b7c4e45346)

</td>
</tr>
</table>

### OrpheusTTS (zh) 模型

<table>
<tr>
<td align="center">

**长乐**
</td>
<td align="center">

**白芷**
</td>
</tr>

<tr>
<td align="center">

[长乐](https://github.com/user-attachments/assets/271bd5a3-dd56-430a-87d0-ebea0e397a89)

</td>
<td align="center">

[白芷](https://github.com/user-attachments/assets/e012c580-840f-401c-99f5-665c72f6b89f)

</td>
</tr>
</table>

## 快速开始

建议在 **python 3.8 - 3.12** 环境中使用 pip 安装 `flashtts`：

```bash
pip install flashtts
```

详细安装步骤，请参考：[installation.md](docs/zh/get_started/installation.md)

本地推理：
```bash
flashtts infer \
  -i "你好，欢迎使用语音合成。" \
  -o output.wav \
  -m ./models/your_model \
  -b vllm \
  [其他可选参数]
```

详细使用说明，请参考：[quick_start.md](docs/zh/get_started/quick_start.md)

部署方式：
```bash
 flashtts serve \
 --model_path Spark-TTS-0.5B \ 
 --backend vllm \ 
 --role_dir data/roles 
 --llm_device cuda \
 --tokenizer_device cuda \
 --detokenizer_device cuda \
 --wav2vec_attn_implementation sdpa \
 --llm_attn_implementation sdpa \ 
 --torch_dtype "bfloat16" \ 
 --max_length 32768 \
 --llm_gpu_memory_utilization 0.6 \
 --host 0.0.0.0 \
 --port 8000
 ```
    
## ⚡ 推理速度

测试环境：`A800 GPU` · 模型：`Spark-TTS‑0.5B` · 测试脚本：[speed_test.py](examples/speed_test.py)

| 场景  |    引擎     | 设备  | 音频长度 (s) | 推理耗时 (s) | RTF  |
|:---:|:---------:|:---:|:--------:|:--------:|:----:|
| 短文本 | llama‑cpp | CPU |   7.48   |   6.81   | 0.91 |
| 短文本 |   torch   | GPU |   7.18   |   7.68   | 1.07 |
| 短文本 |   vllm    | GPU |   7.24   |   1.66   | 0.23 |
| 短文本 |  sglang   | GPU |   7.58   |   1.07   | 0.14 |
| 长文本 | llama‑cpp | CPU |  121.98  |  117.83  | 0.97 |
| 长文本 |   torch   | GPU |  113.70  |  107.17  | 0.94 |
| 长文本 |   vllm    | GPU |  111.82  |   7.28   | 0.07 |
| 长文本 |  sglang   | GPU |  117.02  |   4.20   | 0.04 |

> **RTF < 1** 表示实时合成。

## ⚙️ 使用提示

1. **SparkTTS** 权重需使用 `bfloat16` 或 `float32`；`float16` 会报错。
2. 若遇到长时间空白音，可将 `repetition_penalty` 调高 (> 1.0)。
3. **OrpheusTTS** 支持在文本中插入 `<tag>` 控制情感，标签见 [`orpheus_engine.py`](flashtts/engine/orpheus_engine.py) 中
   `LANG_MAP`。
4. 出于安全考虑，**MegaTTS&nbsp;3** 未公开 WaveVAE
   编码器，请参考官方说明下载：[参考音频](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr)。

## 🤝 鸣谢

- [Spark‑TTS](https://github.com/SparkAudio/Spark-TTS)
- [Orpheus‑TTS](https://github.com/canopyai/Orpheus-TTS)
- [MegaTTS&nbsp;3](https://github.com/bytedance/MegaTTS3)

## ⚠️ 免责声明

FlashTTS 仅供学术研究、教育与合法用途，如无障碍辅助、个性化语音合成等。请勿将其用于欺诈、冒充、深度伪造等违法场景。违者自行承担法律与道德责任。

## License

本项目沿用 Spark‑TTS 的协议，详情参见 [LICENSE](https://github.com/SparkAudio/Spark-TTS/blob/main/LICENSE)。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=FlashTTS/FlashTTS&type=Date)](https://www.star-history.com/#FlashTTS/FlashTTS&Date)