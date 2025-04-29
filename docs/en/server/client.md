# Flash-TTS Python Client Usage Guide

This document provides complete examples for using the Flash-TTS service in Python, including character-based synthesis,
voice cloning (via base64 and file), streaming cloning, and OpenAI SDK-compatible interface.

---

## Environment Setup

1. Install dependencies:
   ```bash
   pip install requests pyaudio openai
   ```
2. Make sure the service is up and accessible, e.g.:
   ```python
   BASE_URL = "http://127.0.0.1:8000"
   ```

---

## 1. Character Synthesis (`/speak`)

- **Endpoint**: `POST {BASE_URL}/speak`
- **Content-Type**: `application/json`

### Request Parameters

| Field             | Type   | Required | Description                         |
|-------------------|--------|----------|-------------------------------------|
| `name`            | string | Yes      | Character name                      |
| `text`            | string | Yes      | Text to be synthesized              |
| `pitch`           | enum   | No       | Pitch (e.g., very_low to very_high) |
| `speed`           | enum   | No       | Speed (e.g., very_low to very_high) |
| `temperature`     | float  | No       | Sampling randomness                 |
| `top_k`           | int    | No       | Top-K sampling                      |
| `top_p`           | float  | No       | Nucleus sampling threshold          |
| `max_tokens`      | int    | No       | Max token count for generation      |
| `stream`          | bool   | No       | Whether to stream the response      |
| `response_format` | enum   | No       | Output format (e.g., mp3/wav)       |

### Sample Code

```python
import requests


def generate_voice():
    payload = {
        "name": "male",
        "text": "Hello, world!",
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
        print("Character voice saved: voice.mp3")
    else:
        print("Error", resp.status_code, resp.text)
```

### Steps

1. Build a JSON request with character name and parameters.
2. Send a POST request.
3. If successful, write the binary response to a file.

---

## 2. Voice Cloning - Base64 Method (`/clone_voice`)

- **Endpoint**: `POST {BASE_URL}/clone_voice`

### Request Parameters

| Field                 | Type   | Required | Description                                                  |
|-----------------------|--------|----------|--------------------------------------------------------------|
| `text`                | string | Yes      | Text to be synthesized                                       |
| `reference_audio`     | string | Yes      | Reference audio in base64 string                             |
| `reference_text`      | string | No       | Text matching the reference audio (optional)                 |
| Other sampling params | ...    | No       | Same as above: `temperature`, `top_k`, `top_p`, `max_tokens` |
| `stream`              | bool   | No       | Whether to stream                                            |
| `response_format`     | enum   | No       | Output format                                                |

### Sample Code

```python
import requests, base64


def clone_with_base64():
    with open("ref.wav", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "text": "Clone this voice",
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
        print("Cloned voice saved: clone.mp3")
```

### Steps

1. Read and base64-encode the local reference audio.
2. Set `reference_audio` in the payload.
3. Send a POST request and save the resulting audio.

---

## 3. Voice Cloning - File Upload Method (`/clone_voice`)

- **Endpoint**: Same as above
- **Content-Type**: `multipart/form-data`

### Request Parameters

- Form fields: `text`, sampling params, etc.
- File fields: `reference_audio_file` (required), optional `latent_file` (for Mega models)

### Sample Code

```python
import requests


def clone_with_file():
    payload = {"text": "Clone audio", "temperature": 0.9}
    files = {"reference_audio_file": open("ref.wav", "rb")}
    resp = requests.post(f"{BASE_URL}/clone_voice", data=payload, files=files)
    if resp.ok:
        open("clone.mp3", "wb").write(resp.content)
        print("File upload clone complete: clone.mp3")
```

### Steps

1. Prepare form data and file handle.
2. Send a multipart POST request.
3. Save the audio output upon success.

---

## 4. Streaming Voice Cloning (`/clone_voice`)

- **Endpoint**: Same as above
- **Settings**: Set `stream=true`, use suitable `response_format` like `wav`

### Sample Code

```python
import requests, base64, pyaudio


def clone_voice_stream():
    with open("ref.wav", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    data = {
        "text": "...",
        "reference_audio": b64,
        "stream": True,
        "response_format": "wav"
    }
    resp = requests.post(f"{BASE_URL}/clone_voice", data=data, stream=True)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
    for chunk in resp.iter_content(1024):
        stream.write(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()
```

### Steps

1. Enable streaming mode in the request.
2. Read the response in chunks and stream playback using PyAudio.

---

## 5. OpenAI-Compatible API (`/v1`)

- **Endpoint**: `POST {BASE_URL}/v1/audio/speech`
- **SDK**: OpenAI Python client

### Sample Code

Call a built-in audio character:

```python
from openai import OpenAI


def openai_speech():
    client = OpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key="not-needed"  # If an API key is set, please provide it
    )
    with client.audio.speech.with_streaming_response.create(
            model="spark",
            voice="赞助商",  # Name of the built-in voice
            input="Hello, I am the invincible little cutie."
    ) as response:
        response.stream_to_file("out.mp3")
    print("Output file: out.mp3")
```

Or provide a reference audio to use the voice cloning feature:

```python
from openai import OpenAI
import base64


def openai_speech():
    client = OpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key="not-needed"  # If an API key is set, please provide it
    )
    with open("data/mega-roles/御姐/御姐配音.wav", "rb") as f:
        audio_bytes = f.read()
    # Convert the binary audio data into a base64-encoded string
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    with client.audio.speech.with_streaming_response.create(
            model="spark",
            voice=audio_base64,  # Replace the 'voice' parameter with the audio's base64 to trigger voice cloning
            input="Hello, I am the invincible little cutie."
    ) as response:
        response.stream_to_file("clone.mp3")
    print("Cloned file: clone.mp3")
```

### Steps

1. Initialize the OpenAI client and set the base URL.
2. Call `audio.speech.create` with streaming enabled.
3. Save the resulting audio file.

---

## 6. Add Speaker (`/add_speaker`)

- **Endpoint**: `POST {BASE_URL}/add_speaker`
- **Content-Type**: `multipart/form-data`

### Example Code

```python
import requests


def add_speaker():
    # Choose to use a URL/base64 or a local file
    files = {
        "audio_file": open("speaker_ref.wav", "rb"),
        "latent_file": open("speaker_latent.npy", "rb")  # If using the Mega engine
    }
    data = {"name": "new_speaker", "reference_text": "Sample audio description"}
    resp = requests.post(f"{BASE_URL}/add_speaker", data=data, files=files)
    if resp.status_code == 200:
        print("Speaker added successfully:", resp.json())
    else:
        print("Failed to add speaker:", resp.status_code, resp.text)
```

---

## 7. Delete Speaker (`/delete_speaker`)

- **Endpoint**: `POST {BASE_URL}/delete_speaker`
- **Content-Type**: `multipart/form-data`

### Example Code

```python
import requests


def delete_speaker():
    data = {"name": "new_speaker"}
    resp = requests.post(f"{BASE_URL}/delete_speaker", data=data)
    if resp.status_code == 200:
        print("Speaker deleted successfully:", resp.json())
    else:
        print("Failed to delete speaker:", resp.status_code, resp.text)
```