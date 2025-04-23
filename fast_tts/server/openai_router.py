# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/7 16:20
# Author  : Hui Huang
from fastapi import HTTPException, Request, APIRouter
from fastapi.responses import JSONResponse, Response, StreamingResponse
from .protocol import OpenAISpeechRequest, ModelCard, ModelList
from .utils.audio_writer import StreamingAudioWriter
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

    try:

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
            pitch=request.pitch,
            speed=request.speed,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            max_tokens=request.max_tokens,
            length_threshold=request.length_threshold,
            window_size=request.window_size
        )
        # Check if streaming is requested (default for OpenAI client)
        if request.stream:
            async def stream_output():
                try:
                    # Stream chunks
                    async for chunk_data in engine.speak_stream_async(
                            **api_inputs
                    ):
                        # Check if client is still connected
                        is_disconnected = client_request.is_disconnected
                        if callable(is_disconnected):
                            is_disconnected = await is_disconnected()
                        if is_disconnected:
                            logger.info("Client disconnected, stopping audio generation")
                            break

                        audio = audio_writer.write_chunk(chunk_data, finalize=False)
                        yield audio
                    yield audio_writer.write_chunk(finalize=True)

                except Exception as e:
                    logger.error(f"Error in single output streaming: {e}")
                    audio_writer.close()
                    raise

            # Standard streaming without download link
            return StreamingResponse(
                stream_output(),
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

            # Generate complete audio using public interface
            audio_data = await engine.speak_async(
                **api_inputs
            )
            output = audio_writer.write_chunk(audio_data, finalize=False)
            final = audio_writer.write_chunk(finalize=True)
            output = output + final
            return Response(
                content=output,
                media_type=content_type,
                headers=headers,
            )

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in speech generation: {str(e)}")
        try:
            audio_writer.close()
        except:
            pass
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )
