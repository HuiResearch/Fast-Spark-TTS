## Flash-TTS 后端部署与接口使用文档

### 1. 安装与启动

1. 参考安装文档: [installation.md](../get_started/installation.md)
2. 启动服务：
   ```bash

    flashtts serve \
    --model_path Spark-TTS-0.5B \ # 可修改为自己的模型地址
    --backend vllm \ # vllm、sglang、torch、llama-cpp、mlx-lm任选一个
    --llm_device cuda \
    --tokenizer_device cuda \
    --detokenizer_device cuda \
    --wav2vec_attn_implementation sdpa \
    --llm_attn_implementation sdpa \ # 如果使用torch engine，最好开启加速
    --torch_dtype "bfloat16" \ # 对于spark-tts模型，不支持bfloat16的设备，只能设置为float32.
    --max_length 32768 \
    --llm_gpu_memory_utilization 0.6 \
    --host 0.0.0.0 \
    --port 8000
    
    ```
3. 在浏览器中访问页面

  ```
  http://localhost:8000
  ```

4. 查看api文档

  ```
  http://localhost:8000/docs
  ```

### 2. 启动参数说明

| 参数                              | 类型    | 描述                                                                                             | 默认值                     |
|---------------------------------|-------|------------------------------------------------------------------------------------------------|-------------------------|
| `--model_path`                  | str   | 必填，TTS 模型目录路径                                                                                  | —                       |
| `--backend`                     | str   | 必填，合成引擎类型，可选：`llama-cpp`, `vllm`, `sglang`, `torch`, `mlx-lm`                                  | —                       |
| `--snac_path`                   | str   | `OrpheusTTS` 的 `SNAC` 模块路径，仅当 `model` 为 `orpheus` 时使用                                          | None                    |
| `--role_dir`                    | str   | 加载角色音频参考目录：Spark 引擎 默认 `data/roles`，Mega 引擎 默认 `data/mega-roles`                               | Spark: `data/roles`     |
|                                 |       |                                                                                                | Mega: `data/mega-roles` |
| `--api_key`                     | str   | API 访问密钥，启用后所有请求需在 `Authorization: Bearer <KEY>` 中携带                                           | None                    |
| `--llm_device`                  | str   | LLM 运行设备，如 `cpu`、`cuda`                                                                        | `auto`                  |
| `--tokenizer_device`            | str   | 音频 `tokenizer` 设备                                                                              | `auto`                  |
| `--detokenizer_device`          | str   | 音频 `detokenizer` 设备                                                                            | `auto`                  |
| `--wav2vec_attn_implementation` | str   | `spark-tts`模型的`wav2vec`模块注意力实现方式，可选：`sdpa`, `flash_attention_2`, `eager`                       | `eager`                 |
| `--llm_attn_implementation`     | str   | `backend`为`torch`时，LLM 注意力实现方式，可选：`sdpa`, `flash_attention_2`, `eager`                         | `eager`                 |
| `--max_length`                  | int   | LLM 最大上下文数量                                                                                    | 32768                   |
| `--llm_gpu_memory_utilization`  | float | `vllm`/`sglang` 显存占用比例                                                                         | 0.6                     |
| `--torch_dtype`                 | str   | 模型精度类型，可选：`float16`, `bfloat16`, `float32`, `auto`                                             | `auto`                  |
| `--cache_implementation`        | str   | `backend`为`torch`时，生成缓存实现：`static`, `offloaded_static`, `sliding_window`, `hybrid`, `mamba`... | None                    |
| `--seed`                        | int   | 随机种子                                                                                           | 0                       |
| `--batch_size`                  | int   | 音频处理组件最大批次大小                                                                                   | 1                       |
| `--llm_batch_size`              | int   | LLM 最大批次大小                                                                                     | 256                     |
| `--wait_timeout`                | float | 动态批处理请求超时秒数                                                                                    | 0.01                    |
| `--host`                        | str   | 服务监听地址                                                                                         | `0.0.0.0`               |
| `--port`                        | int   | 服务监听端口                                                                                         | 8000                    |

### 3. 接口使用流程

示例（cURL）：

```bash
curl -X POST http://localhost:8000/clone_voice \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "text=你好，世界" \
  -F "reference_audio_file=@/path/to/ref.wav" \
  -F "stream=false" \
  -F "response_format=wav" \
  --output output.wav
```

### 4. 请求协议与接口参数

#### 4.1 语音克隆：`POST /clone_voice`

- **Content-Type**: `multipart/form-data`
- **参数说明**：

| 字段                     | 类型      | 必填 | 描述                                                       |
|------------------------|---------|----|----------------------------------------------------------|
| `text`                 | string  | 是  | 待合成文本                                                    |
| `reference_audio`      | string  | 否  | 引用音频，可传 URL 或 base64 编码字符串。与 `reference_audio_file` 二选一。 |
| `reference_audio_file` | file    | 否  | 引用音频文件上传（WAV/MP3 等）。与 `reference_audio` 二选一。             |
| `reference_text`       | string  | 否  | 引用音频对应文本                                                 |
| `pitch`                | enum    | 否  | 音高：`very_low`, `low`, `moderate`, `high`, `very_high`    |
| `speed`                | enum    | 否  | 语速：`very_low`, `low`, `moderate`, `high`, `very_high`    |
| `temperature`          | float   | 否  | 随机性系数，越高生成越多样                                            |
| `top_k`                | int     | 否  | Top-K 采样数                                                |
| `top_p`                | float   | 否  | Nucleus 采样阈值                                             |
| `repetition_penalty`   | float   | 否  | 重复惩罚系数                                                   |
| `max_tokens`           | int     | 否  | 最大生成 token 数                                             |
| `length_threshold`     | int     | 否  | 文本分段阈值，超过后按 `window_size` 分段                             |
| `window_size`          | int     | 否  | 分段窗口大小                                                   |
| `stream`               | boolean | 否  | 是否流式返回（`true`）或一次性返回（`false`）                            |
| `response_format`      | enum    | 否  | 返回音频格式：`mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`        |

#### 4.2 角色合成：`POST /speak`

- **Content-Type**: `application/json`
- **请求体**：

```json
{
  "name": "角色名",
  "text": "待合成文本",
  "pitch": "moderate",
  "speed": "moderate",
  "temperature": 0.9,
  "top_k": 50,
  "top_p": 0.95,
  "repetition_penalty": 1.0,
  "max_tokens": 4096,
  "length_threshold": 50,
  "window_size": 50,
  "stream": false,
  "response_format": "mp3"
}
```

- **字段说明**：与 CloneRequest 相同，额外字段 `name` 指定角色。

#### 4.3 多角色对话合成：`POST /multi_speak`

- **Content-Type**: `application/json`
- **请求体示例**：

```json
{
  "text": "<role:female> 你好！<role:male> 我很好，谢谢！",
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95,
  "repetition_penalty": 1.0,
  "max_tokens": 4096,
  "length_threshold": 50,
  "window_size": 50,
  "stream": true,
  "response_format": "wav"
}
```

- **字段说明**：省略 `name`，通过文本前缀 `<role:角色名>` 确定发言者。

#### 4.4 OpenAI 兼容接口（前缀 `/v1`）

- 路径与功能与上述接口一致，使用 `OpenAISpeechRequest` 协议：
    - `model`: 模型 ID 或名称
    - `input`: 合成文本
    - `voice`: 您想要使用的音频字符的名称，或者参考音频的 URL 或 base64。
    - 其他参数同 Clone/Speak。

#### 4.5 获取角色列表：`GET /audio_roles` 或 `GET /v1/audio_roles`

- **响应示例**：
  ```json
  {
    "success": true,
    "roles": ["alice", "bob", "tara"]
  }
  ```

#### 4.6 添加角色：`POST /add_speaker`

- **Content-Type**: `multipart/form-data`
- **参数说明**：

| 字段               | 类型     | 必填 | 描述                                                |
|------------------|--------|----|---------------------------------------------------|
| `name`           | string | 是  | 要添加的角色名称                                          |
| `audio`          | string | 否  | 引用音频样本 URL 或 base64 编码字符串，与 `audio_file` 二选一      |
| `reference_text` | string | 否  | 与引用音频对应的文本描述或转录                                   |
| `audio_file`     | file   | 否  | 上传引用音频文件（WAV），与 `audio` 二选一                       |
| `latent_file`    | file   | 否  | Mega 引擎使用的 latent 文件（与 `audio`/`audio_file` 组合使用） |

- **响应示例**：
  ```json
  {
    "success": true,
    "role": "角色名"
  }
  ```

#### 4.7 删除角色：`POST /delete_speaker`

- **Content-Type**: `multipart/form-data`
- **参数说明**：

| 字段     | 类型     | 必填 | 描述       |
|--------|--------|----|----------|
| `name` | string | 是  | 要删除的角色名称 |

- **响应示例**：
  ```json
  {
    "success": true,
    "role": "角色名"
  }
  ```
