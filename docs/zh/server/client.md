# Flash-TTS Python 客户端使用指南

本文档提供在 Python 中调用 Flash-TTS 服务的完整示例，包括角色合成、语音克隆（base64 和文件方式）、流式克隆和 OpenAI SDK 兼容接口。

---

## 环境准备

1. 安装依赖：
   ```bash
   pip install requests pyaudio openai
   ```
2. 确保服务已启动并可访问，如：
   ```bash
   BASE_URL = "http://127.0.0.1:8000"
   ```

---

## 1. 角色合成（/speak）

- **接口地址**：`POST {BASE_URL}/speak`
- **Content-Type**：`application/json`

### 请求参数

| 字段                | 类型     | 必填 | 说明                     |
|-------------------|--------|----|------------------------|
| `name`            | string | 是  | 角色名                    |
| `text`            | string | 是  | 待合成文本                  |
| `pitch`           | enum   | 否  | 音高（very_low…very_high） |
| `speed`           | enum   | 否  | 语速（very_low…very_high） |
| `temperature`     | float  | 否  | 随机性系数                  |
| `top_k`           | int    | 否  | Top-K 采样               |
| `top_p`           | float  | 否  | Nucleus 采样阈值           |
| `max_tokens`      | int    | 否  | 最大生成 token 数           |
| `stream`          | bool   | 否  | 是否流式返回                 |
| `response_format` | enum   | 否  | 输出格式（mp3/wav 等）        |

### 示例代码

```python
import requests


def generate_voice():
    payload = {
        "name": "male",
        "text": "你好，世界！",
        "pitch": "moderate",
        "speed": "moderate",
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 0.95,
        "max_tokens": 2048,
        "stream": False,
        "response_format": "mp3"
    }
    resp = requests.post(f"{BASE_URL}/speak", json=payload)
    if resp.status_code == 200:
        with open("voice.mp3", "wb") as f:
            f.write(resp.content)
        print("角色音频已保存：voice.mp3")
    else:
        print("Error", resp.status_code, resp.text)
```

### 步骤说明

1. 构造 JSON 请求体并指定角色与参数。
2. 发送 POST 请求。
3. 请求成功后将二进制内容写入文件。

---

## 2. 语音克隆 - Base64 方式（/clone_voice）

- **接口地址**：`POST {BASE_URL}/clone_voice`

### 请求参数

| 字段                | 类型     | 必填 | 说明                                            |
|-------------------|--------|----|-----------------------------------------------|
| `text`            | string | 是  | 待合成文本                                         |
| `reference_audio` | string | 是  | 参考音频 Base64 字符串                               |
| `reference_text`  | string | 否  | 参考音频对应文本（可选）                                  |
| 其他采样参数同上          | ...    | 否  | `temperature`, `top_k`, `top_p`, `max_tokens` |
| `stream`          | bool   | 否  | 是否流式                                          |
| `response_format` | enum   | 否  | 输出格式                                          |

### 示例代码

```python
import requests, base64


def clone_with_base64():
    with open("ref.wav", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "text": "克隆这段声音",
        "reference_audio": b64,
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 0.95,
        "max_tokens": 2048,
        "stream": False,
        "response_format": "mp3"
    }
    resp = requests.post(f"{BASE_URL}/clone_voice", data=payload)
    if resp.ok:
        open("clone.mp3", "wb").write(resp.content)
        print("克隆音频已保存：clone.mp3")
```

### 步骤说明

1. 将本地参考音频读入并 Base64 编码。
2. 在表单数据中设置 `reference_audio`。
3. POST 请求后将音频保存。

---

## 3. 语音克隆 - 文件上传方式（/clone_voice）

- **接口地址**：同上
- **Content-Type**：`multipart/form-data`

### 请求参数

- 表单字段：`text`, 采样参数等
- 文件字段：`reference_audio_file` (必填)，可选 `latent_file`（Mega 模型时）

### 示例代码

```python
import requests


def clone_with_file():
    payload = {"text": "克隆音频", "temperature": 0.9}
    files = {"reference_audio_file": open("ref.wav", "rb")}
    resp = requests.post(f"{BASE_URL}/clone_voice", data=payload, files=files)
    if resp.ok:
        open("clone.mp3", "wb").write(resp.content)
        print("文件方式克隆完成：clone.mp3")
```

### 步骤说明

1. 构造表单数据与文件句柄。
2. 发送 multipart POST。
3. 成功后写文件。

---

## 4. 流式语音克隆（/clone_voice）

- **接口地址**：同上
- **需设置**：`stream=true`, `response_format` 合适如 `wav`

### 示例代码

```python
import requests, base64, pyaudio


def clone_voice_stream():
    with open("ref.wav", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    data = {"text": "...", "reference_audio": b64, "stream": True, "response_format": "wav"}
    resp = requests.post(f"{BASE_URL}/clone_voice", data=data, stream=True)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
    for chunk in resp.iter_content(1024):
        stream.write(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()
```

### 步骤说明

1. 在表单启用流模式。
2. 分块读取响应，将 PCM 数据实时写入 PyAudio 播放。

---

## 5. OpenAI 兼容接口（/v1）

- **接口地址**：`POST {BASE_URL}/v1/audio/speech`
- **使用 SDK**：OpenAI Python 客户端

### 示例代码

调用内置的音频角色
```python
from openai import OpenAI


def openai_speech():
    client = OpenAI(
       base_url=f"{BASE_URL}/v1",
       api_key="not-needed"  # 如果设置了api key，请传入
    )
    with client.audio.speech.with_streaming_response.create(
            model="spark",
            voice="赞助商",
            input="你好，我是无敌的小可爱。"
    ) as response:
        response.stream_to_file("out.mp3")
    print("输出文件：out.mp3")
```
或者传入参考音频，调用语音克隆功能

```python
from openai import OpenAI
import base64


def openai_speech():
   client = OpenAI(
      base_url=f"{BASE_URL}/v1",
      api_key="not-needed"  # 如果设置了api key，请传入
   )
   with open("data/mega-roles/御姐/御姐配音.wav", "rb") as f:
      audio_bytes = f.read()
   # 将二进制音频数据转换为 base64 字符串
   audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

   with client.audio.speech.with_streaming_response.create(
           model="spark",
           voice=audio_base64,  # 使用音频的base64编码替换voice，即可触发语音克隆
           input="你好，我是无敌的小可爱。"
   ) as response:
      response.stream_to_file("clone.mp3")
   print("克隆文件：out.mp3")
```
### 步骤说明

1. 初始化 OpenAI 客户端，指定 base_url。
2. 调用 `audio.speech.create` 并接收流。
3. 保存输出。

---

## 6. 添加角色（/add_speaker）

- **接口地址**：`POST {BASE_URL}/add_speaker`
- **Content-Type**：`multipart/form-data`

### 示例代码

```python
import requests


def add_speaker():
    # 选择使用 URL/base64 或本地文件
    files = {"audio_file": open("speaker_ref.wav", "rb"),
             "latent_file": open("speaker_latent.npy", "rb")}  # 如果使用 Mega 引擎
    data = {"name": "new_speaker", "reference_text": "示例音频描述"}
    resp = requests.post(f"{BASE_URL}/add_speaker", data=data, files=files)
    if resp.status_code == 200:
        print("添加角色成功：", resp.json())
    else:
        print("添加角色失败：", resp.status_code, resp.text)
```

---

## 7. 删除角色（/delete_speaker）

- **接口地址**：`POST {BASE_URL}/delete_speaker`
- **Content-Type**：`multipart/form-data`

### 示例代码

```python
import requests


def delete_speaker():
    data = {"name": "new_speaker"}
    resp = requests.post(f"{BASE_URL}/delete_speaker", data=data)
    if resp.status_code == 200:
        print("删除角色成功：", resp.json())
    else:
        print("删除角色失败：", resp.status_code, resp.text)
```

