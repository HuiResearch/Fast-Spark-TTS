<div align="center">
  <img src="docs/zh/_img/icon.png" width="450"/>

[📘Documentation](docs/zh/README.MD)

[中文](README.MD) | [English](README_EN.MD)

</div>

______________________________________________________________________

## 介绍

> 基于SparkTTS、OrpheusTTS、MegaTTS3等模型，提供高质量中文语音合成与声音克隆服务。通过简单易用的Web界面，让您轻松创建自然逼真的人声，满足各种场景需求。

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

## 🖼️ 前端展示

https://github.com/user-attachments/assets/1bd9d586-fac7-4016-b955-5a58d8fb9d7e

## 语音 Demos

这里是`Spark-TTS`模型的克隆语音试听案例

---

<table>
<tr>
<td align="center">

**Donald Trump (en)**
</td>
<td align="center">

**Donald Trump (zh)**
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

---

这里是`MegaTTS3`模型的克隆语音试听案例

---

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

---

这里是`OrpheusTTS zh`模型的合成语音试听案例

---

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

---

## 推理速度

- 显卡：`A800`
- 模型：`Spark-TTS-0.5B`
- 测试使用参数和评测方式可参考[speed_test.py](examples/speed_test.py)
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

## 提示

1. `SparkTTS`模型权重量化为`float16`无法使用，`torch_dtype`请设置为`bfloat16`或`float32`。
2. 如果出现长空白音，可尝试调整`repetition_penalty`参数，大于`1.0`会降低重复token生成的概率，可避免长时间静音`token`生成(
   SparkTTS的静音token id为`163406`)。
3. `OrpheusTTS`支持emotion tag，直接在文本中添加`<tag>`
   即可实现。每个模型支持的tag标签请参考：[orpheus_engine.py](fast_tts/engine/orpheus_engine.py) 中 LANG_MAP
4. 出于安全考虑，`MegaTTS3`
   团队未上传WaveVAE编码器参数。因此，只能从此处链接下载参考音频进行推理：[参考音频](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr)
   。如果需要使用自己的音频，请参考MegaTTS3项目的说明：[MegaTTS3](https://github.com/bytedance/MegaTTS3/tree/main?tab=readme-ov-file#inference)

## Acknowledgement

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