# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/7 16:20
# Author  : Hui Huang
from fastapi import HTTPException, Request, APIRouter
from fastapi.responses import JSONResponse, Response, StreamingResponse
from .protocol import OpenAISpeechRequest, ModelCard, ModelList
from .utils.audio_writer import StreamingAudioWriter
from .utils.utils import generate_audio, generate_audio_stream
from ..engine import AutoEngine
from ..logger import get_logger

logger = get_logger()

openai_router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)


@openai_router.get("/models")
async def list_models(raw_request: Request):
    """List all available models"""
    engine: AutoEngine = raw_request.app.state.engine

    # Create standard model list
    models = ModelList(data=[
        ModelCard(id=engine.engine_name)
    ])

    return JSONResponse(content=models.model_dump())


@openai_router.get("/audio/voices")
async def list_voices(raw_request: Request):
    """List all available voices for text-to-speech"""
    engine: AutoEngine = raw_request.app.state.engine
    try:
        voices = engine.list_roles()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve voice list",
                "type": "server_error",
            },
        )


def float_to_speed_label(x: float) -> str:
    """
    将 0.25–4.0 的浮点值映射到 ["very_low","low","moderate","high","very_high"]。

      ≤0.5   → very_low
      (0.5,1)→ low
      (1.0, 2.0 )  → moderate
      (2.0,3.0)→ high
      ≥3   → very_high
    """
    # 1. clamp 到 [0.0,4.0]
    x = max(0.25, min(4.0, x))

    # 2. 根据阈值返回对应的字符串
    if x <= 0.5:
        return "very_low"
    elif x < 1.0:
        return "low"
    elif x < 2.0:
        return "moderate"
    elif x < 3.0:
        return "high"
    else:
        return "very_high"


@openai_router.post("/audio/speech")
async def create_speech(
        request: OpenAISpeechRequest,
        client_request: Request
):
    engine: AutoEngine = client_request.app.state.engine
    if request.model not in [engine.engine_name]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model",
                "message": f"Unsupported model: {request.model}",
                "type": "invalid_request_error",
            },
        )
    audio_writer = StreamingAudioWriter(request.response_format, sample_rate=engine.SAMPLE_RATE)

    # Set content type based on format
    content_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(request.response_format, f"audio/{request.response_format}")

    api_inputs = dict(
        name=request.voice,
        text=request.input,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        max_tokens=request.max_tokens,
        length_threshold=request.length_threshold,
        window_size=request.window_size
    )
    if engine.engine_name.lower() == 'spark':
        api_inputs['pitch'] = float_to_speed_label(request.pitch)
        api_inputs['speed'] = float_to_speed_label(request.speed)
    if request.stream:
        return StreamingResponse(
            generate_audio_stream(
                engine.speak_stream_async,
                api_inputs,
                audio_writer,
                client_request
            ),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )
    else:
        headers = {
            "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
            "Cache-Control": "no-cache",  # Prevent caching
        }
        try:
            # Generate complete audio using public interface
            audio_data = await engine.speak_async(
                **api_inputs
            )
        except Exception as e:
            logger.warning(f"Voice synthesis for the role failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        output = generate_audio(audio_data, audio_writer)

        return Response(
            content=output,
            media_type=content_type,
            headers=headers,
        )
