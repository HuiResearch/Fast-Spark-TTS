# 项目文档总览

欢迎使用`FlashTTS`文档。本项目分为三个主要部分：**快速上手（get_started）**、**模型推理（inference）** 和 **服务器部署（server）**。

---

## 📦 目录结构

```
├── get_started
│   ├── installation.md      # 安装指南
│   └── quick_start.md       # 使用指南
├── inference
│   ├── auto-engine.md       # 自动推理引擎说明
│   ├── mega-tts.md          # Mega TTS 模型文档
│   ├── orpheus-tts.md       # Orpheus TTS 模型文档
│   └── spark-tts.md         # Spark TTS 模型文档
└── server
    ├── client.md            # 客户端使用说明
    └── server.md            # 服务端部署说明
```

## 🛠️ 快速开始

- [安装指南](get_started/installation.md)：了解如何安装项目依赖，并进行基础配置，快速启动。
- [使用指南](get_started/quick_start.md)：快速使用命令行进行语音合成。

## 🧠 推理引擎与模型

- [mega-tts](inference/mega-tts.md)：Mega TTS 模型的配置和使用方法。
- [orpheus-tts](inference/orpheus-tts.md)：Orpheus TTS 模型的详细文档。
- [spark-tts](inference/spark-tts.md)：Spark TTS 模型的推理流程与特点。
- [auto-engine](inference/auto-engine.md)：介绍项目中的自动推理引擎模块。

## 🌐 部署服务

- [server](server/server.md)：部署服务端的步骤，包括运行环境与部署命令。
- [client](server/client.md)：如何调用模型 API，包括示例请求与返回格式。

