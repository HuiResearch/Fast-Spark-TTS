# -*- coding: utf-8 -*-
# Time      :2025/3/29 15:36
# Author    :Hui Huang
import asyncio
import os.path
import re
from typing import Literal, Optional, Callable, AsyncIterator

import numpy as np
import torch

from .base_engine import BaseEngine
from ..audio import SnacDeTokenizer
from ..logger import get_logger
from .utils import limit_concurrency

logger = get_logger()

LANG_MAP = {
    "mandarin": {
        "voices": ["长乐", "白芷"],
        "tags": ["嬉笑", "轻笑", "呻吟", "大笑", "咳嗽", "抽鼻子", "咳"],
        "default": "长乐"
    },
    "french": {
        "voices": ["pierre", "amelie", "marie"],
        "tags": ["chuckle", "cough", "gasp", "groan", "laugh", "sigh", "sniffle", "whimper", "yawn"],
        "default": "pierre"
    },
    "german": {
        "voices": ["jana", "thomas", "max"],
        "tags": ["chuckle", "cough", "gasp", "groan", "laugh", "sigh", "sniffle", "yawn"],
        "default": "jana"
    },
    "korean": {
        "voices": ["유나", "준서"],
        "tags": ["한숨", "헐", "헛기침", "훌쩍", "하품", "낄낄", "신음", "작은 웃음", "기침", "으르렁"],
        "default": "유나"
    },
    "hindi": {
        "voices": ["ऋतिका"],
        "tags": [],
        "default": "ऋतिका"
    },
    "spanish": {
        "voices": ["javi", "sergio", "maria"],
        "tags": ["groan", "chuckle", "gasp", "resoplido", "laugh", "yawn", "cough"],
        "default": "javi"
    },
    "italian": {
        "voices": ["pietro", "giulia", "carlo"],
        "tags": ["sigh", "laugh", "cough", "sniffle", "groan", "yawn", "gemito", "gasp"],
        "default": "pietro"
    },
    "english": {
        "voices": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
        "tags": ["laugh", "chuckle", "sigh", "cough", "sniffle", "groan", "yawn", "gasp"],
        "default": "tara"
    }
}


class AsyncOrpheusEngine(BaseEngine):
    SAMPLE_RATE = 24000

    def __init__(
            self,
            model_path: str,
            max_length: int = 8192,
            lang: Literal[
                "mandarin",
                "french",
                "german",
                "korean",
                "hindi",
                "spanish",
                "italian",
                "spanish_italian",
                "english",
                None
            ] = None,
            snac_path: Optional[str] = None,
            llm_device: Literal["cpu", "cuda", "mps", "auto"] | str = "auto",
            detokenizer_device: Literal["cpu", "cuda", "mps", "auto"] | str = "auto",
            backend: Literal["vllm", "llama-cpp", "sglang", "torch", "mlx-lm"] = "torch",
            llm_attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            torch_dtype: Literal['float16', "bfloat16", 'float32', 'auto'] = "auto",
            llm_gpu_memory_utilization: Optional[float] = 0.8,  # snac模型显存暂用很小
            cache_implementation: Optional[str] = None,
            batch_size: int = 1,
            llm_batch_size: int = 256,
            wait_timeout: float = 0.01,
            seed: int = 0,
            **kwargs
    ):
        self.lang = self._auto_detect_lang(model_path, lang)
        self.seed = seed
        self.set_seed(seed)
        self.detokenizer = SnacDeTokenizer(
            snac_path if snac_path is not None else os.path.join(model_path, "snac"),
            device=self._auto_detect_device(detokenizer_device),
            batch_size=batch_size,
            wait_timeout=wait_timeout)
        if self.lang == "spanish_italian":
            self.speakers = set(LANG_MAP["spanish"]['voices'] + LANG_MAP["italian"]['voices'])
            self.speakers = list(self.speakers)
            self.speakers.sort()

            self.tags = set(LANG_MAP["spanish"]['tags'] + LANG_MAP["italian"]['tags'])
            self.tags = list(self.tags)
            self.tags.sort()
            self.default_speaker = LANG_MAP["spanish"]["default"]
        else:
            self.speakers = LANG_MAP[self.lang]["voices"]
            self.tags = LANG_MAP[self.lang]["tags"]

            self.default_speaker = LANG_MAP[self.lang]["default"]

        super().__init__(
            llm_model_path=model_path,
            max_length=max_length,
            llm_device=llm_device,
            backend=backend,
            llm_attn_implementation=llm_attn_implementation,
            torch_dtype=torch_dtype,
            llm_gpu_memory_utilization=llm_gpu_memory_utilization,
            cache_implementation=cache_implementation,
            llm_batch_size=llm_batch_size,
            seed=seed,
            stop_token_ids=[128258, 128262],
            **kwargs
        )

    def _auto_detect_lang(
            self,
            model_path: str,
            lang: Literal[
                "mandarin",
                "french",
                "german",
                "korean",
                "hindi",
                "spanish",
                "italian",
                "spanish_italian",
                "english",
                None
            ] = "english"
    ) -> str:
        if model_path is not None and model_path.endswith("/"):
            model_path = model_path[:-1]
        model_name = os.path.split(model_path)[-1]
        if "zh" in model_name:
            detect_lang = "mandarin"
        elif "hi" in model_name:
            detect_lang = "hindi"
        elif "ko" in model_name:
            detect_lang = "korean"
        elif "fr" in model_name:
            detect_lang = "french"
        elif "de" in model_name:
            detect_lang = "german"
        elif "es_it" in model_name:
            detect_lang = "spanish_italian"
        else:
            detect_lang = None

        if lang is not None:
            if detect_lang is not None and detect_lang != lang:
                if lang in ["spanish", "italian"] and detect_lang == "spanish_italian":
                    pass
                else:
                    logger.warning(
                        f"{model_name} detected language is {detect_lang}, but you set `lang` to {lang}. `lang` will be corrected to `{detect_lang}`.")
                    lang = detect_lang
            elif detect_lang is None:
                logger.info(f"`lang` will be set to `{lang}`.")
        else:
            if detect_lang is None:
                logger.warning(
                    f"`lang` will be set to `english`.")
                lang = "english"
            else:
                logger.info(f"{model_name} detected language is {detect_lang}. `lang` will be set to `{detect_lang}`")
                lang = detect_lang
        return lang

    def list_roles(self) -> list[str]:
        roles = list(self.speakers)
        roles.sort()
        return roles

    def apply_prompt(
            self,
            text: str,
            name: Optional[str] = None
    ):
        if name is None:
            name = self.default_speaker
        if name not in self.speakers:
            err_msg = f"{name} is not in the currently supported speaker list. Currently supported speakers are: {', '.join(self.speakers)}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        prompt = f"<custom_token_3><|begin_of_text|>{name}: {text}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
        return prompt

    async def _convert_to_audio(self, multiframe: list[int]) -> np.ndarray | None:
        if len(multiframe) < 28:
            return None

        num_frames = len(multiframe) // 7
        # 截取完整帧的数据
        frame = multiframe[: num_frames * 7]
        # 将列表转换为 torch 张量，并重塑为 (num_frames, 7) 的形状
        frame_tensor = torch.tensor(frame, dtype=torch.int32).view(num_frames, 7)

        # 分别提取各个通道的 tokens
        # codes_0: 每帧的第 0 个元素，形状为 (num_frames,)
        codes_0 = frame_tensor[:, 0]
        # codes_1: 每帧的第 1 和第 4 个元素，形状为 (num_frames, 2) 后展平为 (num_frames*2,)
        codes_1 = frame_tensor[:, [1, 4]].reshape(-1)
        # codes_2: 每帧的第 2、3、5、6 个元素，形状为 (num_frames, 4) 后展平为 (num_frames*4,)
        codes_2 = frame_tensor[:, [2, 3, 5, 6]].reshape(-1)

        # 添加 batch 维度，使得形状分别变为 (1, num_frames)，(1, num_frames*2) 和 (1, num_frames*4)
        codes_0 = codes_0.unsqueeze(0)
        codes_1 = codes_1.unsqueeze(0)
        codes_2 = codes_2.unsqueeze(0)

        # 检查所有 token 是否均在 [0, 4096] 范围内
        if ((codes_0 < 0).any() or (codes_0 > 4096).any() or
                (codes_1 < 0).any() or (codes_1 > 4096).any() or
                (codes_2 < 0).any() or (codes_2 > 4096).any()):
            return None

        audio_hat = await self.detokenizer.detokenize_async([codes_0, codes_1, codes_2])
        # Process output
        audio = audio_hat["audio"][:, :, 2048:4096].detach().cpu().numpy()
        audio = (audio * 32767).astype(np.int16).reshape(1, -1)
        return audio.squeeze(0)

    async def _speak_stream(
            self,
            prompt: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            **kwargs
    ) -> AsyncIterator[np.ndarray]:
        buffer = []
        index = 0
        pattern = re.compile(r"<custom_token_(\d+)>")
        async for text_token in self.generator.async_stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                **kwargs
        ):
            text_token = text_token.text
            audio_ids = pattern.findall(text_token)
            for audio_id in audio_ids:
                audio_id = int(audio_id)
                audio_id = int(audio_id) - 10 - ((index % 7) * 4096)

                if audio_id > 0:
                    buffer.append(audio_id)
                    index += 1

                    # Convert to audio when we have enough tokens
                    if index % 7 == 0 and index > 27:
                        buffer_to_proc = buffer[-28:]
                        audio_samples = await self._convert_to_audio(buffer_to_proc)
                        if audio_samples is not None:
                            yield audio_samples

    async def _speak(
            self,
            prompt: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            **kwargs) -> np.ndarray:
        buffer = []
        async for chunk in self._speak_stream(
                prompt=prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                **kwargs
        ):
            buffer.append(chunk)
        return np.concatenate(buffer, axis=0)

    async def speak_stream_async(
            self,
            text: str,
            name: Optional[str] = None,
            pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
            speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> AsyncIterator[np.ndarray]:
        if pitch is not None or speed is not None:
            logger.warning("OrpheusTTS does not support adjusting pitch and speed.")
        self.set_seed(seed=self.seed)
        segments = self.preprocess_text(
            text=text,
            length_threshold=length_threshold,
            window_size=window_size,
            split_fn=split_fn
        )
        prompts = [self.apply_prompt(name=name, text=seg) for seg in segments]
        pre_buffer = np.array([], dtype=np.int16)
        pre_buffer_size = self.SAMPLE_RATE * kwargs.get("audio_chunk_duration", 1.5)
        started_playback = False
        for prompt in prompts:
            async for audio in self._speak_stream(
                    prompt=prompt,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    **kwargs
            ):
                if not started_playback:
                    pre_buffer = np.concatenate([pre_buffer, audio], axis=0)
                    if pre_buffer.shape[0] >= pre_buffer_size:
                        started_playback = True
                        yield pre_buffer
                else:
                    yield audio
        if not started_playback:
            yield pre_buffer

    async def speak_async(
            self,
            text: str,
            name: Optional[str] = None,
            pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
            speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs) -> np.ndarray:
        if pitch is not None or speed is not None:
            logger.warning("OrpheusTTS does not support adjusting pitch and speed.")
        self.set_seed(seed=self.seed)
        segments = self.preprocess_text(
            text=text,
            length_threshold=length_threshold,
            window_size=window_size,
            split_fn=split_fn
        )
        prompts = [self.apply_prompt(name=name, text=seg) for seg in segments]

        semaphore = asyncio.Semaphore(self._batch_size)  # 限制并发数，避免超长文本卡死
        limit_speak = limit_concurrency(semaphore)(self._speak)
        tasks = [
            asyncio.create_task(
                limit_speak(
                    prompt=prompt,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    **kwargs
                )
            ) for prompt in prompts
        ]
        # 并发执行所有任务
        audios = await asyncio.gather(*tasks)
        final_audio = np.concatenate(audios, axis=0)
        return final_audio
