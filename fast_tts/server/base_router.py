# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/7 16:14
# Author  : Hui Huang
import base64
import io
from typing import Optional, Annotated, Literal

import httpx
import numpy as np
from fastapi import HTTPException, Request, APIRouter, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse, JSONResponse, Response, FileResponse
from .protocol import CloneRequest, SpeakRequest, MultiSpeakRequest
from .utils.audio_writer import StreamingAudioWriter
from ..engine import AutoEngine
from ..logger import get_logger

logger = get_logger()

base_router = APIRouter(
    tags=["Fast-TTS"],
    responses={404: {"description": "Not found"}},
)


async def generate_audio_stream(generator, data, writer: StreamingAudioWriter, raw_request: Request):
    async for chunk in generator(**data):
        # Check if client is still connected
        is_disconnected = raw_request.is_disconnected
        if callable(is_disconnected):
            is_disconnected = await is_disconnected()
        if is_disconnected:
            logger.info("Client disconnected, stopping audio generation")
            break

        audio = writer.write_chunk(chunk, finalize=False)
        yield audio
    yield writer.write_chunk(finalize=True)


async def generate_audio(audio: np.ndarray, writer: StreamingAudioWriter):
    output = writer.write_chunk(audio, finalize=False)
    final = writer.write_chunk(finalize=True)
    output = output + final
    return output


@base_router.get("/")
async def get_web():
    return FileResponse("templates/index.html")


@base_router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("templates/favicon.ico")


async def get_audio_bytes_from_url(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="无法从指定 URL 下载参考音频")
        return response.content


def parse_clone_form(
        text: str = Form(...),
        reference_audio: Optional[str] = Form(None),
        reference_text: Optional[str] = Form(None),
        pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Form(None),
        speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Form(None),
        temperature: float = Form(0.9),
        top_k: int = Form(50),
        top_p: float = Form(0.95),
        repetition_penalty: float = Form(1.0),
        max_tokens: int = Form(4096),
        length_threshold: int = Form(50),
        window_size: int = Form(50),
        stream: bool = Form(False),
        response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Form("mp3"),
):
    return CloneRequest(
        text=text,
        reference_audio=reference_audio,
        reference_text=reference_text,
        pitch=pitch,
        speed=speed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        length_threshold=length_threshold,
        window_size=window_size,
        stream=stream,
        response_format=response_format,
    )


# 克隆语音接口：接收 multipart/form-data，上传参考音频和其它表单参数
@base_router.post("/clone_voice")
async def clone_voice(
        req: Annotated[CloneRequest, Depends(parse_clone_form)],
        raw_request: Request,
        reference_audio_file: Optional[UploadFile] = File(None),
        latent_file: Optional[UploadFile] = File(None),
):
    engine: AutoEngine = raw_request.app.state.engine
    if engine.engine_name == 'orpheus':
        logger.error("OrpheusTTS 暂不支持语音克隆.")
        raise HTTPException(status_code=500, detail="OrpheusTTS 暂不支持该功能.")

    if reference_audio_file is None:
        # 根据 reference_audio 内容判断读取方式
        if req.reference_audio.startswith("http://") or req.reference_audio.startswith("https://"):
            audio_bytes = await get_audio_bytes_from_url(req.reference_audio)
        else:
            try:
                audio_bytes = base64.b64decode(req.reference_audio)
            except Exception as e:
                logger.warning("无效的 base64 音频数据: " + str(e))
                raise HTTPException(status_code=400, detail="无效的 base64 音频数据: " + str(e))
        # 利用 BytesIO 包装字节数据，然后使用 soundfile 读取为 numpy 数组
        try:
            bytes_io = io.BytesIO(audio_bytes)
        except Exception as e:
            logger.warning("读取参考音频失败: " + str(e))
            raise HTTPException(status_code=400, detail="读取参考音频失败: " + str(e))
    else:
        content = await reference_audio_file.read()
        if not content:
            logger.warning("参考音频文件为空")
            raise HTTPException(status_code=400, detail="参考音频文件为空")
        bytes_io = io.BytesIO(content)

    if engine.engine_name == 'mega':
        if latent_file is None:
            err_msg = "MegaTTS克隆音频需要上传参考音频的latent_file(.npy)。"
            logger.warning(err_msg)
            raise HTTPException(status_code=400, detail=err_msg)
        else:
            contents = await latent_file.read()
            latent_io = io.BytesIO(contents)
        reference_audio = (bytes_io, latent_io)
    else:
        reference_audio = bytes_io

    audio_writer = StreamingAudioWriter(req.response_format, sample_rate=engine.SAMPLE_RATE)
    # Set content type based on format
    content_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(req.response_format, f"audio/{req.response_format}")

    if req.stream:
        data = dict(
            text=req.text,
            reference_audio=reference_audio,
            reference_text=req.reference_text,
            pitch=req.pitch,
            speed=req.speed,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_tokens=req.max_tokens,
            length_threshold=req.length_threshold,
            window_size=req.window_size
        )
        return StreamingResponse(
            generate_audio_stream(
                engine.clone_voice_stream_async,
                data,
                audio_writer,
                raw_request
            ),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )
    else:
        try:
            audio = await engine.clone_voice_async(
                text=req.text,
                reference_audio=reference_audio,
                reference_text=req.reference_text,
                pitch=req.pitch,
                speed=req.speed,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                max_tokens=req.max_tokens,
                length_threshold=req.length_threshold,
                window_size=req.window_size,
            )
        except Exception as e:
            logger.warning(f"克隆语音失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        headers = {
            "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
            "Cache-Control": "no-cache",  # Prevent caching
        }
        audio_io = await generate_audio(audio, writer=audio_writer)
        return Response(
            audio_io,
            media_type=content_type,
            headers=headers,
        )


@base_router.get("/audio_roles")
async def audio_roles(raw_request: Request):
    roles = raw_request.app.state.engine.list_roles()
    return JSONResponse(
        content={
            "success": True,
            "roles": roles
        })


@base_router.post("/speak")
async def speak(req: SpeakRequest, raw_request: Request):
    engine: AutoEngine = raw_request.app.state.engine
    if req.name not in engine.list_roles():
        err_msg = f"{req.name} 不在已有的角色列表中。"
        logger.warning(err_msg)
        raise HTTPException(status_code=500, detail=err_msg)

    audio_writer = StreamingAudioWriter(req.response_format, sample_rate=engine.SAMPLE_RATE)
    # Set content type based on format
    content_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(req.response_format, f"audio/{req.response_format}")

    if req.stream:
        data = dict(
            name=req.name,
            text=req.text,
            temperature=req.temperature,
            pitch=req.pitch,
            speed=req.speed,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_tokens=req.max_tokens,
            length_threshold=req.length_threshold,
            window_size=req.window_size
        )
        return StreamingResponse(
            generate_audio_stream(
                engine.speak_stream_async,
                data,
                audio_writer,
                raw_request
            ),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )
    else:
        try:
            audio = await engine.speak_async(
                name=req.name,
                text=req.text,
                pitch=req.pitch,
                speed=req.speed,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                max_tokens=req.max_tokens,
                length_threshold=req.length_threshold,
                window_size=req.window_size,
            )
        except Exception as e:
            logger.warning(f"角色语音合成失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        headers = {
            "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
            "Cache-Control": "no-cache",  # Prevent caching
        }
        audio_io = await generate_audio(audio, writer=audio_writer)
        return Response(
            audio_io,
            media_type=content_type,
            headers=headers,
        )


@base_router.post("/multi_speak")
async def multi_speak(req: MultiSpeakRequest, raw_request: Request):
    engine: AutoEngine = raw_request.app.state.engine

    audio_writer = StreamingAudioWriter(req.response_format, sample_rate=engine.SAMPLE_RATE)
    # Set content type based on format
    content_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(req.response_format, f"audio/{req.response_format}")

    if req.stream:
        data = dict(
            text=req.text,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_tokens=req.max_tokens,
            length_threshold=req.length_threshold,
            window_size=req.window_size
        )
        return StreamingResponse(
            generate_audio_stream(
                engine.multi_speak_stream_async,
                data,
                audio_writer,
                raw_request
            ),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )
    else:
        try:
            audio = await engine.multi_speak_async(
                text=req.text,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                max_tokens=req.max_tokens,
                length_threshold=req.length_threshold,
                window_size=req.window_size,
            )
        except Exception as e:
            logger.warning(f"多角色语音合成失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        headers = {
            "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
            "Cache-Control": "no-cache",  # Prevent caching
        }
        audio_io = await generate_audio(audio, writer=audio_writer)
        return Response(
            audio_io,
            media_type=content_type,
            headers=headers,
        )
