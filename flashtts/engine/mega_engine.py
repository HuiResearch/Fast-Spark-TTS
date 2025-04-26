# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/17 12:06
# Author  : Hui Huang
import asyncio
import os
from typing import Literal, Optional, Callable

import numpy as np
import torch

from .base_engine import BaseEngine
from .utils import limit_concurrency
from ..audio import MegaTokenizer
from ..modules.mega_modules.ph_tone_convert import split_ph
from ..logger import get_logger

logger = get_logger()


class AsyncMega3Engine(BaseEngine):
    SAMPLE_RATE = 24000
    _SUPPORT_CLONE = True
    _SUPPORT_SPEAK = True

    def __init__(
            self,
            model_path: str,
            max_length: int = 32768,
            llm_device: Literal["cpu", "cuda", "mps", "auto"] | str = "auto",
            tokenizer_device: Literal["cpu", "cuda", "mps", "auto"] | str = "auto",
            backend: Literal["vllm", "llama-cpp", "sglang", "torch", "mlx-lm"] = "torch",
            llm_attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            torch_dtype: Literal['float16', "bfloat16", 'float32', 'auto'] = "auto",
            llm_gpu_memory_utilization: Optional[float] = 0.6,
            cache_implementation: Optional[str] = None,
            batch_size: int = 1,
            llm_batch_size: int = 256,
            seed: int = 0,
            **kwargs):
        self.seed = seed
        self.set_seed(seed)
        self.audio_tokenizer = MegaTokenizer(
            model_path,
            device=self._auto_detect_device(tokenizer_device)
        )
        self._batch_size = batch_size
        super().__init__(
            llm_model_path=os.path.join(model_path, "g2p"),
            max_length=max_length,
            llm_device=llm_device,
            backend=backend,
            llm_attn_implementation=llm_attn_implementation,
            torch_dtype=torch_dtype,
            llm_gpu_memory_utilization=llm_gpu_memory_utilization,
            cache_implementation=cache_implementation,
            llm_batch_size=llm_batch_size,
            stop_token_ids=[152468],
            seed=seed,
            **kwargs
        )
        self.speech_start_idx = self.generator.tokenizer.encode("<Reserved_TTS_0>")[0]
        self.speakers = {}

    def list_roles(self) -> list[str]:
        roles = list(self.speakers.keys())
        roles.sort()
        return roles

    def apply_prompt(self, text: str) -> list[int]:
        prompt = '<BOT>' + text + '<BOS>'
        txt_token = self.generator.tokenizer(prompt)['input_ids']
        input_ids = txt_token + [145 + self.speech_start_idx]
        return input_ids

    async def _generate(
            self,
            text: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            **kwargs):
        prompt = self.apply_prompt(text)
        generated_output = await self.generator.async_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=False,
            **kwargs
        )
        outputs = generated_output.token_ids
        # process outputs
        ph_tokens = [
            token_id - self.speech_start_idx
            for token_id in outputs if
            (token_id >= self.speech_start_idx and token_id not in self.generator.stop_token_ids)]
        ph_pred, tone_pred = split_ph(ph_tokens)
        return {
            "ph_pred": torch.LongTensor(ph_pred)[None, :],
            "tone_pred": torch.LongTensor(tone_pred)[None, :]
        }

    def combine_audio_segments(self, segments, crossfade_duration=0.16):
        window_length = int(self.SAMPLE_RATE * crossfade_duration)
        hanning_window = np.hanning(2 * window_length)
        # Combine
        for i, segment in enumerate(segments):
            if i == 0:
                combined_audio = segment
            else:
                overlap = (
                        combined_audio[-window_length:] * hanning_window[window_length:]
                        + segment[:window_length] * hanning_window[:window_length]
                )
                combined_audio = np.concatenate(
                    [combined_audio[:-window_length], overlap, segment[window_length:]]
                )
        return combined_audio

    async def _clone_voice_from_ref(
            self,
            text: str,
            resource_context,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            time_step: int = 32,
            p_w: float = 1.6,
            t_w: float = 2.5,
            **kwargs
    ) -> np.ndarray:
        segments = self.preprocess_text(
            text,
            window_size=window_size,
            split_fn=split_fn,
            length_threshold=length_threshold
        )
        semaphore = asyncio.Semaphore(self._batch_size)  # 限制并发数，避免超长文本卡死
        limit_generate = limit_concurrency(semaphore)(self._generate)
        tasks = [
            asyncio.create_task(limit_generate(
                segment,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                **kwargs)) for segment in segments
        ]
        # 并发执行所有任务
        generate_results = await asyncio.gather(*tasks)

        pred_wav_list = []

        for seg_i, generate_result in enumerate(generate_results):
            mel2ph_pred = self.audio_tokenizer.dur_predict(
                ctx_dur_tokens=resource_context['ctx_dur_tokens'],
                incremental_state_dur_prompt=resource_context['incremental_state_dur_prompt'],
                ph_pred=generate_result['ph_pred'],
                tone_pred=generate_result['tone_pred'],
                dur_disturb=kwargs.get('dur_disturb', 0.1),
                dur_alpha=kwargs.get('dur_alpha', 1.0),
                is_first=(seg_i == 0),
                is_final=(seg_i == len(segments) - 1),
            )
            wav_pred = self.audio_tokenizer.decode(
                mel2ph_ref=resource_context['mel2ph_ref'],
                mel2ph_pred=mel2ph_pred,
                ph_ref=resource_context['ph_ref'],
                tone_ref=resource_context['tone_ref'],
                ph_pred=generate_result['ph_pred'],
                tone_pred=generate_result['tone_pred'],
                vae_latent=resource_context['vae_latent'],
                loudness_prompt=resource_context['loudness_prompt'],
                time_step=time_step,
                p_w=p_w,
                t_w=t_w,
            )
            pred_wav_list.append(wav_pred)

        audio = self.combine_audio_segments(pred_wav_list).astype(float)
        audio = (audio * 32767).astype(np.int16)
        return audio

    async def clone_voice_async(
            self,
            text: str,
            reference_audio: tuple,
            reference_text: Optional[str] = None,
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
            time_step: int = 32,
            p_w: float = 1.6,
            t_w: float = 2.5,
            **kwargs) -> np.ndarray:
        self.set_seed(seed=self.seed)
        assert len(
            reference_audio) == 2, "The reference audio for MegaTTS3 requires two files to be provided: a WAV audio file and an encoded NPY file."
        if pitch is not None or speed is not None:
            logger.warning("MegaTTS does not support adjusting pitch and speed.")
        resource_context = self.audio_tokenizer.preprocess(
            audio=reference_audio[0],
            latent_file=reference_audio[1],
        )
        audio = await self._clone_voice_from_ref(
            text=text,
            resource_context=resource_context,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            length_threshold=length_threshold,
            window_size=window_size,
            split_fn=split_fn,
            time_step=time_step,
            p_w=p_w,
            t_w=t_w,
            **kwargs
        )
        return audio

    async def add_speaker(self, name: str, audio: tuple, reference_text: Optional[str] = None):
        if name in self.speakers:
            logger.warning(f"{name} 音频已存在，将使用新的音频覆盖。")
        assert len(
            audio) == 2, "The reference audio for MegaTTS3 requires two files to be provided: a WAV audio file and an encoded NPY file."

        resource_context = self.audio_tokenizer.preprocess(
            audio=audio[0],
            latent_file=audio[1],
        )
        self.speakers[name] = resource_context

    async def delete_speaker(self, name: str):
        if name not in self.speakers:
            logger.warning(f"{name} 角色不存在。")
            return
        self.speakers.pop(name)

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
            time_step: int = 32,
            p_w: float = 1.6,
            t_w: float = 2.5,
            **kwargs) -> np.ndarray:
        if pitch is not None or speed is not None:
            logger.warning("MegaTTS does not support adjusting pitch and speed.")
        if name is None and len(self.speakers) > 0:
            name = list(self.speakers.keys())[0]
        if name not in self.speakers:
            err_msg = f"{name} 角色不存在。"
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.set_seed(seed=self.seed)
        speaker = self.speakers[name]
        audio = await self._clone_voice_from_ref(
            text=text,
            resource_context=speaker,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            length_threshold=length_threshold,
            window_size=window_size,
            split_fn=split_fn,
            time_step=time_step,
            p_w=p_w,
            t_w=t_w,
            **kwargs
        )
        return audio
