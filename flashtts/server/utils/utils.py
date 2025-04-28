# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/25 09:47
# Author  : Hui Huang
import base64
import io

import httpx
import numpy as np
from fastapi import HTTPException

from .audio_writer import StreamingAudioWriter
from ...logger import get_logger

logger = get_logger()


async def get_audio_bytes_from_url(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="无法从指定 URL 下载参考音频")
        return response.content


async def load_audio_bytes(audio_file, audio):
    if audio_file is None:
        # 根据 reference_audio 内容判断读取方式
        if audio.startswith("http://") or audio.startswith("https://"):
            audio_bytes = await get_audio_bytes_from_url(audio)
        else:
            try:
                audio_bytes = base64.b64decode(audio)
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
        content = await audio_file.read()
        if not content:
            logger.warning("参考音频文件为空")
            raise HTTPException(status_code=400, detail="参考音频文件为空")
        bytes_io = io.BytesIO(content)
    return bytes_io


async def load_latent_file(latent_file):
    if latent_file is None:
        err_msg = "MegaTTS克隆音频需要上传参考音频的latent_file(.npy)。"
        logger.warning(err_msg)
        raise HTTPException(status_code=400, detail=err_msg)
    else:
        contents = await latent_file.read()
        latent_io = io.BytesIO(contents)
    return latent_io


async def generate_audio_stream(generator, data, writer: StreamingAudioWriter, raw_request):
    try:
        async for chunk in generator(**data):
            # Check if client is still connected
            if await raw_request.is_disconnected():
                logger.info("Client disconnected, stopping audio generation")
                break

            audio = writer.write_chunk(chunk, finalize=False)
            yield audio
        yield writer.write_chunk(finalize=True)
    except Exception as e:
        try:
            writer.close()
        except:
            pass
        logger.error(f"An error occurred during the streaming audio generation: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"An error occurred during the streaming audio generation: {str(e)}")
    finally:
        try:
            writer.close()
        except:
            pass


def generate_audio(audio: np.ndarray, writer: StreamingAudioWriter):
    try:
        output = writer.write_chunk(audio, finalize=False)
        final = writer.write_chunk(finalize=True)
        output = output + final
    except Exception as e:
        try:
            writer.close()
        except:
            pass
        logger.error(f"An error occurred while writing the audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while writing the audio: {str(e)}")
    try:
        writer.close()
    except:
        pass
    return output
