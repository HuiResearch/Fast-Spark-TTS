# Project Documentation Overview

Welcome to the `FastTTS` documentation. This project is divided into three main sections: **Getting
Started (`get_started`)**, **Model Inference (`inference`)**, and **Server Deployment (`server`)**.

---

## 📦 Directory Structure

```
├── get_started
│   └── installation.md      # Installation Guide
├── inference
│   ├── auto-engine.md       # Auto Inference Engine Guide
│   ├── mega-tts.md          # Mega TTS Model Documentation
│   ├── orpheus-tts.md       # Orpheus TTS Model Documentation
│   └── spark-tts.md         # Spark TTS Model Documentation
└── server
    ├── client.md            # Client Usage Guide
    └── server.md            # Server Deployment Guide
```

## 🛠️ Getting Started

- [Installation Guide](get_started/installation.md): Learn how to install project dependencies, perform basic setup, and
  get started quickly.

## 🧠 Inference Engines and Models

- [mega-tts](inference/mega-tts.md): Configuration and usage of the Mega TTS model.
- [orpheus-tts](inference/orpheus-tts.md): Detailed documentation for the Orpheus TTS model.
- [spark-tts](inference/spark-tts.md): Inference process and features of the Spark TTS model.
- [auto-engine](inference/auto-engine.md): Introduction to the project's automatic inference engine module.

## 🌐 Deploying the Service

- [server](server/server.md): Steps to deploy the server, including runtime environment and deployment commands.
- [client](server/client.md): How to call the model APIs, including example requests and response formats.