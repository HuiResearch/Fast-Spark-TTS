# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/7 12:10
# Author  : Hui Huang
import os
from typing import Optional, Literal, Callable, AsyncIterator
import numpy as np
from .base_engine import Engine
from ..logger import get_logger

logger = get_logger()

_ENGINE_DIR_NAMES = {
    "spark": [
        "LLM",
        "BiCodec",
        "wav2vec2-large-xlsr-53"
    ],
    "orpheus": [
        "snac"
    ],
    "mega": [
        "aligner_lm",
        "diffusion_transformer",
        "duration_lm",
        "g2p",
        "wavvae"
    ]
}


class AutoEngine(Engine):
    def __init__(
            self,
            model_path: str,
            max_length: int = 32768,
            snac_path: Optional[str] = None,
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
            llm_device: Literal["cpu", "cuda", "mps", "auto"] | str = "auto",
            tokenizer_device: Literal["cpu", "cuda", "mps", "auto"] | str = "auto",
            detokenizer_device: Literal["cpu", "cuda", "mps", "auto"] | str = "auto",
            backend: Literal["vllm", "llama-cpp", "sglang", "torch", "mlx-lm"] = "torch",
            wav2vec_attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            llm_attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            torch_dtype: Literal['float16', "bfloat16", 'float32', 'auto'] = "auto",
            llm_gpu_memory_utilization: Optional[float] = 0.6,
            cache_implementation: Optional[str] = None,
            batch_size: int = 1,
            llm_batch_size: int = 256,
            wait_timeout: float = 0.01,
            seed: int = 0,
            **kwargs,
    ):
        engine_name = self._auto_detect_engine(model_path, snac_path)

        engine_kwargs = dict(
            model_path=model_path,
            max_length=max_length,
            llm_device=llm_device,
            backend=backend,
            llm_attn_implementation=llm_attn_implementation,
            torch_dtype=torch_dtype,
            llm_gpu_memory_utilization=llm_gpu_memory_utilization,
            cache_implementation=cache_implementation,
            batch_size=batch_size,
            llm_batch_size=llm_batch_size,
            seed=seed,
            **kwargs,
        )
        if engine_name == 'spark':
            from .spark_engine import AsyncSparkEngine

            engine_cls = AsyncSparkEngine
            engine_kwargs.update(
                {
                    "tokenizer_device": tokenizer_device,
                    "detokenizer_device": detokenizer_device,
                    "wav2vec_attn_implementation": wav2vec_attn_implementation,
                    "wait_timeout": wait_timeout
                }
            )

        elif engine_name == 'orpheus':
            from .orpheus_engine import AsyncOrpheusEngine

            engine_cls = AsyncOrpheusEngine
            engine_kwargs.update(
                {
                    "snac_path": snac_path,
                    "lang": lang,
                    "detokenizer_device": detokenizer_device,
                    "wait_timeout": wait_timeout
                }
            )
        elif engine_name == 'mega':
            from .mega_engine import AsyncMega3Engine

            engine_cls = AsyncMega3Engine
            engine_kwargs.update(
                {
                    "tokenizer_device": tokenizer_device,
                }
            )
        else:
            raise RuntimeError(f"Unknown engine '{engine_name}'")
        config_str = [f"{k}={v!r}" for k, v in engine_kwargs.items()]
        logger.info(
            f"Initializing `AutoEngine(engine={engine_name})` with config: ({', '.join(config_str)})")
        self._engine = engine_cls(**engine_kwargs)
        self.SAMPLE_RATE = self._engine.SAMPLE_RATE
        self.engine_name = engine_name

    @classmethod
    def _auto_detect_engine(cls, model_path: str, snac_path: Optional[str] = None):
        dirs = os.listdir(model_path)
        if all(name in dirs for name in _ENGINE_DIR_NAMES["spark"]):
            return "spark"
        elif all(name in dirs for name in _ENGINE_DIR_NAMES["mega"]):
            return "mega"
        elif "snac" in dirs or snac_path is not None:
            return "orpheus"
        else:
            raise RuntimeError("No engine found")

    def write_audio(self, audio: np.ndarray, filepath: str):
        self._engine.write_audio(audio, filepath)

    def list_roles(self) -> list[str]:
        return self._engine.list_roles()

    async def add_speaker(self, name: str, audio, reference_text: Optional[str] = None):
        await self._engine.add_speaker(name, audio, reference_text=reference_text)

    async def delete_speaker(self, name: str):
        await self._engine.delete_speaker(name)

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
        parameters = dict(
            name=name,
            text=text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            length_threshold=length_threshold,
            window_size=window_size,
            split_fn=split_fn,
            **kwargs
        )
        if self.engine_name == "spark":
            parameters.update(
                {
                    "pitch": pitch,
                    "speed": speed,
                }
            )
        audio = await self._engine.speak_async(
            **parameters,
        )
        return audio

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
        parameters = dict(
            name=name,
            text=text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            length_threshold=length_threshold,
            window_size=window_size,
            split_fn=split_fn,
            **kwargs
        )
        if self.engine_name == "spark":
            parameters.update(
                {
                    "pitch": pitch,
                    "speed": speed,
                }
            )
        async for chunk in self._engine.speak_stream_async(
                name=name,
                text=text,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                length_threshold=length_threshold,
                window_size=window_size,
                split_fn=split_fn,
                **kwargs
        ):
            yield chunk

    async def clone_voice_async(
            self,
            text: str,
            reference_audio,
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
            **kwargs) -> np.ndarray:
        parameters = dict(
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            length_threshold=length_threshold,
            window_size=window_size,
            split_fn=split_fn,
            **kwargs
        )
        if self.engine_name == "spark":
            parameters.update(
                {
                    "pitch": pitch,
                    "speed": speed,
                }
            )
        audio = await self._engine.clone_voice_async(
            **parameters,
        )
        return audio

    async def clone_voice_stream_async(
            self,
            text: str,
            reference_audio,
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
            **kwargs) -> AsyncIterator[np.ndarray]:
        parameters = dict(
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            length_threshold=length_threshold,
            window_size=window_size,
            split_fn=split_fn,
            **kwargs
        )
        if self.engine_name == "spark":
            parameters.update(
                {
                    "pitch": pitch,
                    "speed": speed,
                }
            )
        async for chunk in self._engine.clone_voice_stream_async(
                **parameters
        ):
            yield chunk

    async def multi_speak_async(
            self,
            text: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs
    ) -> np.ndarray:
        audio = await self._engine.multi_speak_async(
            text=text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            length_threshold=length_threshold,
            window_size=window_size,
            split_fn=split_fn,
            **kwargs
        )
        return audio

    async def multi_speak_stream_async(
            self,
            text: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            length_threshold: int = 50,
            window_size: int = 50,
            split_fn: Optional[Callable[[str], list[str]]] = None,
            **kwargs
    ) -> AsyncIterator[np.ndarray]:
        async for chunk in self._engine.multi_speak_stream_async(
                text=text,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                length_threshold=length_threshold,
                window_size=window_size,
                split_fn=split_fn,
                **kwargs
        ):
            yield chunk
