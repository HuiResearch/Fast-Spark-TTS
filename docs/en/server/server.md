## Fast-TTS Backend Deployment and API Usage Guide

### 1. Installation & Startup

1. Refer to the installation guide: [installation.md](../get_started/installation.md)
2. Start the server:

   ```bash
   python server.py \
   --model_path Spark-TTS-0.5B \ # Change to your model path if needed
   --backend vllm \ # Choose from: vllm, sglang, torch, llama-cpp, mlx-lm
   --llm_device cuda \
   --tokenizer_device cuda \
   --detokenizer_device cuda \
   --wav2vec_attn_implementation sdpa \
   --llm_attn_implementation sdpa \ # Recommended for torch backend
   --torch_dtype "bfloat16" \ # Spark-TTS does not support bfloat16 on all devices; use float32 if needed
   --max_length 32768 \
   --llm_gpu_memory_utilization 0.6 \
   --host 0.0.0.0 \
   --port 8000
   ```

3. Access the web interface:
   ```
   http://localhost:8000
   ```

4. View API documentation:
   ```
   http://localhost:8000/docs
   ```

---

### 2. Server Startup Arguments (`server.py`)

| Argument                        | Type  | Description                                                                                        | Default                 |
|---------------------------------|-------|----------------------------------------------------------------------------------------------------|-------------------------|
| `--model_path`                  | str   | Required. Path to the TTS model directory                                                          | —                       |
| `--backend`                     | str   | Required. TTS backend engine. Options: `llama-cpp`, `vllm`, `sglang`, `torch`, `mlx-lm`            | —                       |
| `--snac_path`                   | str   | Path to OrpheusTTS SNAC module. Required only if model is `orpheus`                                | None                    |
| `--role_dir`                    | str   | Directory for role audio references. Default: `data/roles` for Spark, `data/mega-roles` for Mega   | Spark: `data/roles`     |
|                                 |       |                                                                                                    | Mega: `data/mega-roles` |
| `--api_key`                     | str   | API key for access. All requests must include `Authorization: Bearer <KEY>` if enabled             | None                    |
| `--llm_device`                  | str   | Device for running the LLM (e.g., `cpu`, `cuda`)                                                   | `auto`                  |
| `--tokenizer_device`            | str   | Device for the audio tokenizer                                                                     | `auto`                  |
| `--detokenizer_device`          | str   | Device for the audio detokenizer                                                                   | `auto`                  |
| `--wav2vec_attn_implementation` | str   | Attention implementation for `wav2vec` in Spark-TTS. Options: `sdpa`, `flash_attention_2`, `eager` | `eager`                 |
| `--llm_attn_implementation`     | str   | Attention method for LLM (torch backend). Options: `sdpa`, `flash_attention_2`, `eager`            | `eager`                 |
| `--max_length`                  | int   | Max LLM context length                                                                             | 32768                   |
| `--llm_gpu_memory_utilization`  | float | GPU memory usage ratio (for `vllm`/`sglang`)                                                       | 0.6                     |
| `--torch_dtype`                 | str   | Model precision type. Options: `float16`, `bfloat16`, `float32`, `auto`                            | `auto`                  |
| `--cache_implementation`        | str   | Cache strategy for `torch` backend: `static`, `offloaded_static`, `sliding_window`, etc.           | None                    |
| `--seed`                        | int   | Random seed                                                                                        | 0                       |
| `--batch_size`                  | int   | Max batch size for audio processing                                                                | 1                       |
| `--llm_batch_size`              | int   | Max LLM batch size                                                                                 | 256                     |
| `--wait_timeout`                | float | Timeout (in seconds) for dynamic batching                                                          | 0.01                    |
| `--host`                        | str   | Host address to bind                                                                               | `0.0.0.0`               |
| `--port`                        | int   | Port number to listen on                                                                           | 8000                    |

---

### 3. API Usage Workflow

Example using `cURL`:

```bash
curl -X POST http://localhost:8000/clone_voice \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "text=Hello, world" \
  -F "reference_audio_file=@/path/to/ref.wav" \
  -F "stream=false" \
  -F "response_format=wav" \
  --output output.wav
```

---

### 4. API Endpoints and Parameters

#### 4.1 Voice Cloning: `POST /clone_voice`

- **Content-Type**: `multipart/form-data`
- **Parameters**:

| Field                  | Type    | Required | Description                                                                |
|------------------------|---------|----------|----------------------------------------------------------------------------|
| `text`                 | string  | Yes      | Text to synthesize                                                         |
| `reference_audio`      | string  | No       | Reference audio (URL or base64 string). Use this or `reference_audio_file` |
| `reference_audio_file` | file    | No       | Upload reference audio file (WAV/MP3, etc.)                                |
| `reference_text`       | string  | No       | Transcription of the reference audio                                       |
| `pitch`                | enum    | No       | Pitch: `very_low`, `low`, `moderate`, `high`, `very_high`                  |
| `speed`                | enum    | No       | Speed: `very_low`, `low`, `moderate`, `high`, `very_high`                  |
| `temperature`          | float   | No       | Controls randomness in generation                                          |
| `top_k`                | int     | No       | Top-K sampling                                                             |
| `top_p`                | float   | No       | Nucleus sampling threshold                                                 |
| `repetition_penalty`   | float   | No       | Penalty to reduce repetition                                               |
| `max_tokens`           | int     | No       | Max number of tokens to generate                                           |
| `length_threshold`     | int     | No       | Threshold to split long text                                               |
| `window_size`          | int     | No       | Window size for chunking                                                   |
| `stream`               | boolean | No       | Return streaming audio (`true`) or full audio (`false`)                    |
| `response_format`      | enum    | No       | Output audio format: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`            |

#### 4.2 Role-based Synthesis: `POST /speak`

- **Content-Type**: `application/json`
- **Body Example**:

```json
{
  "name": "RoleName",
  "text": "Text to synthesize",
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

- **Note**: Same fields as CloneRequest, with an additional `name` field for the voice role.

#### 4.3 Multi-Speaker Dialogue Synthesis: `POST /multi_speak`

- **Content-Type**: `application/json`
- **Body Example**:

```json
{
  "text": "<role:female> Hello! <role:male> I'm good, thank you!",
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

- **Note**: The `name` field is omitted; speaker is indicated by the prefix `<role:role_name>` in the text.

#### 4.4 OpenAI-Compatible Endpoint (Prefix `/v1`)

- Paths and functionality mirror the standard API.
- Uses `OpenAISpeechRequest` format:
    - `model`: Model ID or name
    - `input`: Text to synthesize
    - `voice`: Voice name or preset
    - Other parameters same as Clone/Speak

#### 4.5 Retrieve Available Roles: `GET /audio_roles` or `GET /v1/audio_roles`

- **Response Example**:
  ```json
  {
    "success": true,
    "roles": ["alice", "bob", "tara"]
  }
  ```