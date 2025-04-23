# -*- coding: utf-8 -*-
# Time      :2025/3/29 11:16
# Author    :Hui Huang
import asyncio
import math
import os.path
import re
from dataclasses import dataclass
from typing import Literal, Optional, Callable, Tuple, AsyncIterator

import numpy as np
import torch

from .base_engine import BaseEngine
from .utils import limit_concurrency
from ..audio import (
    SparkTokenizer,
    SparkDeTokenizer
)
from ..logger import get_logger

logger = get_logger()

TASK_TOKEN_MAP = {
    "vc": "<|task_vc|>",
    "tts": "<|task_tts|>",
    "asr": "<|task_asr|>",
    "s2s": "<|task_s2s|>",
    "t2s": "<|task_t2s|>",
    "understand": "<|task_understand|>",
    "caption": "<|task_cap|>",
    "controllable_tts": "<|task_controllable_tts|>",
    "prompt_tts": "<|task_prompt_tts|>",
    "speech_edit": "<|task_edit|>",
}

LEVELS_MAP = {
    "very_low": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "very_high": 4,
}

GENDER_MAP = {
    "female": 0,
    "male": 1,
}


@dataclass
class SparkAcousticTokens:
    prompt: str
    global_tokens: Optional[torch.Tensor] = None

    def __post_init__(self):
        self._parse_prompt()

    def _parse_prompt(self):
        acoustic = re.findall(
            r"(<\|start_acoustic_token\|>.*?<\|end_global_token\|>)",
            self.prompt)
        global_tokens = [int(token) for token in re.findall(r"bicodec_global_(\d+)", self.prompt)]
        if len(acoustic) == 0:
            raise ValueError("No acoustic tokens found in prompt")
        else:
            self.prompt = acoustic[0]
        if len(global_tokens) == 0:
            raise ValueError("No global tokens found in prompt")
        else:
            global_token_ids = (
                torch.tensor(global_tokens).unsqueeze(0).long()
            )
            self.global_tokens = global_token_ids

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf8') as w:
            w.write(self.prompt)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r', encoding='utf8') as r:
            prompt = r.read()
        return cls(prompt=prompt)


def process_prompt(
        text: str,
        prompt_text: Optional[str] = None,
        global_token_ids: torch.Tensor = None,
        semantic_token_ids: torch.Tensor = None,
        pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
        speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
) -> Tuple[str, torch.Tensor]:
    """
    Process input for voice cloning.

    Args:
        text: The text input to be converted to speech.
        prompt_text: Transcript of the prompt audio.
        global_token_ids: Global token IDs extracted from reference audio.
        semantic_token_ids: Semantic token IDs extracted from reference audio.
        pitch (str): very_low | low | moderate | high | very_high
        speed (str): very_low | low | moderate | high | very_high

    Returns:
        Tuple containing the formatted input prompt and global token IDs.
    """
    # Convert global tokens to string format
    global_tokens = "".join(
        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
    )

    attribte_tokens = None
    if pitch is not None and speed is not None:
        pitch_level_id = LEVELS_MAP[pitch]
        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_level_id = LEVELS_MAP[speed]
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        attribte_tokens = "".join(
            [pitch_label_tokens, speed_label_tokens]
        )
    audio_text = text if prompt_text is None or len(prompt_text) == 0 else (text + prompt_text)
    inputs = [
        TASK_TOKEN_MAP["tts"],
        "<|start_content|>",
        audio_text,
        "<|end_content|>"
    ]
    if attribte_tokens is not None:
        inputs.extend([
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ])
    inputs.extend([
        "<|start_global_token|>",
        global_tokens,
        "<|end_global_token|>",
    ])
    if prompt_text is not None and len(prompt_text) > 0:
        semantic_tokens = "".join(
            [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
        )
        inputs.extend([
            "<|start_semantic_token|>",
            semantic_tokens,
        ])
    # Join all input components into a single string
    inputs = "".join(inputs)
    return inputs, global_token_ids


def process_prompt_control(
        text: str,
        gender: Optional[Literal["female", "male"]] = "female",
        pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
        speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
):
    """
    Process input for voice creation.

    Args:
        gender (str): female | male.
        pitch (str): very_low | low | moderate | high | very_high
        speed (str): very_low | low | moderate | high | very_high
        text (str): The text input to be converted to speech.

    Return:
        str: Input prompt
    """
    gender = gender or "female"
    pitch = pitch or "moderate"
    speed = speed or "moderate"

    assert gender in GENDER_MAP.keys()
    assert pitch in LEVELS_MAP.keys()
    assert speed in LEVELS_MAP.keys()

    gender_id = GENDER_MAP[gender]
    pitch_level_id = LEVELS_MAP[pitch]
    speed_level_id = LEVELS_MAP[speed]

    pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
    speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
    gender_tokens = f"<|gender_{gender_id}|>"

    attribte_tokens = "".join(
        [gender_tokens, pitch_label_tokens, speed_label_tokens]
    )

    control_tts_inputs = [
        TASK_TOKEN_MAP["controllable_tts"],
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_style_label|>",
        attribte_tokens,
        "<|end_style_label|>",
    ]

    return "".join(control_tts_inputs)


class AsyncSparkEngine(BaseEngine):
    SAMPLE_RATE = 16000

    def __init__(
            self,
            model_path: str,
            max_length: int = 32768,
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
        """

        Args:
            model_path: 权重路径
            max_length: llm上下文最大长度
            gguf_model_file: llama cpp加载gguf模型文件，不传入则默认路径为 "{model_path}/LLM/model.gguf"
            llm_device: llm使用的device
            tokenizer_device: audio tokenizer使用的device
            detokenizer_device: audio detokenizer使用device
            backend: llm 后端类型
            wav2vec_attn_implementation: audio tokenizer中，wav2vec模型使用attn算子
            llm_gpu_memory_utilization: vllm和sglang暂用显存比例，单卡可降低该参数
            batch_size: 音频处理组件单批次处理的最大请求数。
            wait_timeout:
            **kwargs:
        """
        self.seed = seed
        self.set_seed(seed)

        self.audio_tokenizer = SparkTokenizer(
            model_path,
            device=self._auto_detect_device(tokenizer_device),
            attn_implementation=wav2vec_attn_implementation,
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )
        self.audio_detokenizer = SparkDeTokenizer(
            model_path,
            device=self._auto_detect_device(detokenizer_device),
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )
        self.speakers = {"female": {}, "male": {}}

        super().__init__(
            llm_model_path=os.path.join(model_path, "LLM"),
            max_length=max_length,
            llm_device=llm_device,
            backend=backend,
            llm_attn_implementation=llm_attn_implementation,
            torch_dtype=torch_dtype,
            llm_gpu_memory_utilization=llm_gpu_memory_utilization,
            cache_implementation=cache_implementation,
            llm_batch_size=llm_batch_size,
            seed=seed,
            stop_tokens=["<|end_semantic_token|>"],
            **kwargs
        )

    def list_roles(self) -> list[str]:
        roles = list(self.speakers.keys())
        roles.sort()
        return roles

    @classmethod
    def apply_prompt(
            cls,
            text: str,
            prompt_text: Optional[str] = None,
            global_token_ids: torch.Tensor = None,
            semantic_token_ids: torch.Tensor = None,
            gender: Optional[Literal["female", "male"]] = None,
            pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
            speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
    ):
        if global_token_ids is not None and semantic_token_ids is not None:
            return process_prompt(
                text, prompt_text,
                global_token_ids, semantic_token_ids,
                pitch=pitch, speed=speed)
        else:
            return process_prompt_control(text, gender, pitch, speed)

    async def _tokenize(
            self,
            audio
    ):
        output = await self.audio_tokenizer.tokenize_async(audio)
        return output

    async def _generate_audio_tokens(
            self,
            prompt: str,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.0,
            max_tokens: int = 4096,
            **kwargs
    ) -> dict[str, torch.Tensor | str]:
        generated_output = await self.generator.async_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **kwargs
        )
        generated_output = generated_output.text
        pred_semantic_tokens = [int(token) for token in re.findall(r"bicodec_semantic_(\d+)", generated_output)]
        if len(pred_semantic_tokens) == 0:
            err_msg = f"Semantic tokens 预测为空，prompt：{prompt}，llm output：{generated_output}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        pred_semantic_ids = (
            torch.tensor(pred_semantic_tokens).to(torch.int32)
        )

        output = {
            "semantic_tokens": pred_semantic_ids,
            "completion": generated_output,
        }

        global_tokens = [int(token) for token in re.findall(r"bicodec_global_(\d+)", generated_output)]
        if len(global_tokens) > 0:
            global_token_ids = (
                torch.tensor(global_tokens).unsqueeze(0).long()
            )
            output["global_tokens"] = global_token_ids

        return output

    async def tokens2wav(
            self,
            global_tokens: torch.Tensor,
            semantic_tokens: torch.Tensor
    ) -> np.ndarray:
        detokenizer_req = {
            "global_tokens": global_tokens,
            "semantic_tokens": semantic_tokens,
        }
        audio = await self.audio_detokenizer.detokenize_async(
            request=detokenizer_req
        )
        audio = audio["audio"][0].detach().cpu().numpy().astype(np.float32)
        return audio

    async def _clone_voice_by_tokens(
            self,
            text: str,
            global_tokens: torch.Tensor,
            semantic_tokens: torch.Tensor,
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
        segments = self.split_text(
            text,
            window_size=window_size,
            split_fn=split_fn,
            length_threshold=length_threshold
        )

        async def clone_segment(segment):
            prompt, global_token_ids = self.apply_prompt(
                text=segment,
                prompt_text=reference_text,
                global_token_ids=global_tokens,
                semantic_token_ids=semantic_tokens,
                pitch=pitch,
                speed=speed,
            )
            generated = await self._generate_audio_tokens(
                prompt=prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                **kwargs
            )
            return await self.tokens2wav(
                global_tokens=global_tokens,
                semantic_tokens=generated["semantic_tokens"]
            )

        if len(segments) > 1:
            semaphore = asyncio.Semaphore(self._batch_size)  # 限制并发数，避免超长文本卡死
            limit_clone_segment = limit_concurrency(semaphore)(clone_segment)
            tasks = [asyncio.create_task(limit_clone_segment(segment)) for segment in segments]
            # 并发执行所有任务
            audios = await asyncio.gather(*tasks)
            final_audio = np.concatenate(audios, axis=0)
        elif len(segments) == 1:
            final_audio = await clone_segment(segments[0])
        else:
            logger.error(f"请传入有效文本：{segments}")
            raise ValueError(f"请传入有效文本：{segments}")

        return final_audio

    async def _post_process_chunk_audio(
            self,
            audio_idx: int,
            semantic_tokens: list[int],
            chunk_size: int,
            global_tokens: torch.Tensor,
            cross_fade_samples: int,
            fade_in: np.ndarray,
            fade_out: np.ndarray,
            last_chunk_audio: np.ndarray,
            overlap_chunk_size: int
    ):
        chunk_semantic_tokens = semantic_tokens[:chunk_size]
        chunk_semantic_tokens = (
            torch.tensor(chunk_semantic_tokens).to(torch.int32)
        )
        chunk_audio = await self.tokens2wav(
            global_tokens=global_tokens,
            semantic_tokens=chunk_semantic_tokens
        )
        if audio_idx == 0:
            yield_audio = chunk_audio[:-cross_fade_samples]
        else:
            cross_faded_overlap = chunk_audio[:cross_fade_samples] * fade_in + last_chunk_audio[
                                                                               -cross_fade_samples:] * fade_out
            yield_audio = np.concatenate(
                [cross_faded_overlap, chunk_audio[cross_fade_samples:-cross_fade_samples]], axis=0)

        return {
            "yield_audio": yield_audio,
            "audio_idx": audio_idx + 1,
            "last_chunk_audio": chunk_audio,
            "semantic_tokens": semantic_tokens[chunk_size - overlap_chunk_size:]
        }

    async def _clone_voice_stream_by_tokens(
            self,
            text: str,
            global_tokens: torch.Tensor,
            semantic_tokens: torch.Tensor,
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
            audio_chunk_duration: float = 1.0,
            max_audio_chunk_duration: float = 8.0,
            audio_chunk_size_scale_factor: float = 2.0,
            audio_chunk_overlap_duration: float = 0.1,
            **kwargs) -> AsyncIterator[np.ndarray]:
        if audio_chunk_duration < 0.5:
            err_msg = "audio_chunk_duration at least 0.5 seconds"
            logger.error(err_msg)
            raise ValueError(err_msg)
        if audio_chunk_size_scale_factor < 1.0:
            err_msg = "audio_chunk_size_scale_factor should be greater than 1, change it according to your actual rtf"
            logger.error(err_msg)
            raise ValueError(err_msg)

        audio_tokenizer_frame_rate = 50
        max_chunk_size = math.ceil(max_audio_chunk_duration * audio_tokenizer_frame_rate)
        chunk_size = math.ceil(audio_chunk_duration * audio_tokenizer_frame_rate)
        overlap_chunk_size = math.ceil(audio_chunk_overlap_duration * audio_tokenizer_frame_rate)
        cross_fade_samples = int(audio_chunk_overlap_duration * self.SAMPLE_RATE)
        fade_out = np.linspace(1, 0, cross_fade_samples)
        fade_in = np.linspace(0, 1, cross_fade_samples)

        segments = self.split_text(
            text,
            window_size=window_size,
            split_fn=split_fn,
            length_threshold=length_threshold
        )

        prompts = []
        for segment in segments:
            prompt, _ = process_prompt(
                text=segment,
                prompt_text=reference_text,
                global_token_ids=global_tokens,
                semantic_token_ids=semantic_tokens,
                pitch=pitch,
                speed=speed,
            )
            prompts.append(prompt)

        out_semantic_tokens = []
        audio_index = 0
        last_audio = None
        for prompt in prompts:
            async for tokens in self.generator.async_stream_generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    **kwargs
            ):
                tokens = tokens.text
                pred_semantic_tokens = [
                    int(token) for token in re.findall(r"bicodec_semantic_(\d+)", tokens)]

                out_semantic_tokens.extend(pred_semantic_tokens)

                if len(out_semantic_tokens) >= chunk_size:
                    processed = await self._post_process_chunk_audio(
                        audio_idx=audio_index,
                        semantic_tokens=out_semantic_tokens,
                        chunk_size=chunk_size,
                        global_tokens=global_tokens,
                        cross_fade_samples=cross_fade_samples,
                        fade_in=fade_in,
                        fade_out=fade_out,
                        last_chunk_audio=last_audio,
                        overlap_chunk_size=overlap_chunk_size
                    )

                    yield processed['yield_audio']

                    audio_index = processed['audio_idx']
                    last_audio = processed["last_chunk_audio"]
                    out_semantic_tokens = processed["semantic_tokens"]
                    # increase chunk size for better speech quality
                    chunk_size = min(max_chunk_size, int(chunk_size * audio_chunk_size_scale_factor))

        if len(out_semantic_tokens) > 0:
            processed = await self._post_process_chunk_audio(
                audio_idx=audio_index,
                semantic_tokens=out_semantic_tokens,
                chunk_size=len(out_semantic_tokens),
                global_tokens=global_tokens,
                cross_fade_samples=cross_fade_samples,
                fade_in=fade_in,
                fade_out=fade_out,
                last_chunk_audio=last_audio,
                overlap_chunk_size=overlap_chunk_size
            )

            yield processed['yield_audio']
            last_audio = processed['last_chunk_audio']
        if last_audio is not None:
            yield last_audio[-cross_fade_samples:]

    async def add_speaker(self, name: str, audio, reference_text: Optional[str] = None):
        if name in self.speakers:
            logger.warning(f"{name} 音频已存在，将使用新的音频覆盖。")

        if name not in ["female", "male"]:
            tokens = await self._tokenize(
                audio
            )
            self.speakers[name] = {
                "global_tokens": tokens['global_tokens'].detach().cpu(),
                "semantic_tokens": tokens['semantic_tokens'].detach().cpu(),
                "reference_text": reference_text
            }
        else:
            self.speakers[name] = {}

    async def delete_speaker(self, name: str):
        if name not in self.speakers:
            logger.warning(f"{name} 角色不存在。")
            return
        self.speakers.pop(name)

    async def _control_generate(
            self,
            text: str,
            gender: Optional[Literal["female", "male"]] = "female",
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
            acoustic_tokens: Optional[SparkAcousticTokens | str] = None,
            return_acoustic_tokens: bool = False,
            **kwargs):
        segments = self.split_text(
            text,
            window_size=window_size,
            split_fn=split_fn,
            length_threshold=length_threshold
        )

        async def generate_audio(
                segment: str,
                acoustic_token: Optional[SparkAcousticTokens] = None
        ):
            prompt = self.apply_prompt(text=segment, gender=gender, pitch=pitch, speed=speed)
            if acoustic_token is not None:
                prompt += acoustic_token.prompt
            generated = await self._generate_audio_tokens(
                prompt=prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                **kwargs
            )
            current_global_tokens = (
                generated['global_tokens'] if acoustic_token is None else acoustic_token.global_tokens)

            audio = await self.tokens2wav(
                global_tokens=current_global_tokens,
                semantic_tokens=generated['semantic_tokens']
            )
            return {
                "audio": audio,
                "completion": generated['completion']
            }

        if acoustic_tokens is not None and isinstance(acoustic_tokens, str):
            acoustic_tokens = SparkAcousticTokens(acoustic_tokens)

        audios = []
        if acoustic_tokens is None:
            # 如果没有传入音色，使用第一段生成音色token，将其与后面片段一起拼接，使用相同音色token引导输出semantic tokens。
            first_output = await generate_audio(segments[0], acoustic_token=None)
            acoustic_tokens = SparkAcousticTokens(first_output['completion'])
            audios.append(first_output['audio'])
            segments = segments[1:]

        if len(segments) > 0:
            semaphore = asyncio.Semaphore(self._batch_size)  # 限制并发数，避免超长文本卡死
            limit_generate_audio = limit_concurrency(semaphore)(generate_audio)

            tasks = [asyncio.create_task(
                limit_generate_audio(
                    segment,
                    acoustic_token=acoustic_tokens
                )
            ) for segment in segments]
            # 并发执行所有任务
            generated_segments = await asyncio.gather(*tasks)
            audios = audios + [out['audio'] for out in generated_segments]
        final_audio = np.concatenate(audios, axis=0)
        if return_acoustic_tokens:
            return final_audio, acoustic_tokens
        return final_audio

    async def speak_async(
            self,
            text: str,
            name: str = "female",
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
            acoustic_tokens: Optional[SparkAcousticTokens | str] = None,
            return_acoustic_tokens: bool = False,
            **kwargs) -> np.ndarray | tuple[np.ndarray, SparkAcousticTokens]:
        name = name or "female"
        if name not in self.speakers:
            err_msg = f"{name} 角色不存在。"
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.set_seed(seed=self.seed)
        speaker = self.speakers[name]
        acoustic_tokens = None
        if name in ["female", "male"]:
            output = await self._control_generate(
                text=text,
                gender=name,
                pitch=pitch,
                speed=speed,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                length_threshold=length_threshold,
                window_size=window_size,
                split_fn=split_fn,
                acoustic_tokens=acoustic_tokens,
                return_acoustic_tokens=return_acoustic_tokens,
                **kwargs
            )
            if return_acoustic_tokens and isinstance(output, tuple):
                audio = output[0]
                acoustic_tokens = output[1]
            else:
                audio = output
        else:
            audio = await self._clone_voice_by_tokens(
                text=text,
                global_tokens=speaker['global_tokens'],
                semantic_tokens=speaker['semantic_tokens'],
                reference_text=speaker['reference_text'],
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
        audio = (audio * 32767).astype(np.int16)
        if acoustic_tokens is not None:
            return audio, acoustic_tokens
        return audio

    async def _control_stream_generate(
            self,
            text: str,
            gender: Optional[Literal["female", "male"]] = "female",
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
            audio_chunk_duration: float = 1.0,
            max_audio_chunk_duration: float = 8.0,
            audio_chunk_size_scale_factor: float = 2.0,
            audio_chunk_overlap_duration: float = 0.1,
            acoustic_tokens: Optional[SparkAcousticTokens | str] = None,
            return_acoustic_tokens: bool = False,
            **kwargs
    ):
        if audio_chunk_duration < 0.5:
            err_msg = "audio_chunk_duration at least 0.5 seconds"
            logger.error(err_msg)
            raise ValueError(err_msg)
        if audio_chunk_size_scale_factor < 1.0:
            err_msg = "audio_chunk_size_scale_factor should be greater than 1, change it according to your actual rtf"
            logger.error(err_msg)
            raise ValueError(err_msg)

        if acoustic_tokens is not None and isinstance(acoustic_tokens, str):
            acoustic_tokens = SparkAcousticTokens(acoustic_tokens)

        audio_tokenizer_frame_rate = 50
        max_chunk_size = math.ceil(max_audio_chunk_duration * audio_tokenizer_frame_rate)
        chunk_size = math.ceil(audio_chunk_duration * audio_tokenizer_frame_rate)
        overlap_chunk_size = math.ceil(audio_chunk_overlap_duration * audio_tokenizer_frame_rate)
        cross_fade_samples = int(audio_chunk_overlap_duration * self.SAMPLE_RATE)
        fade_out = np.linspace(1, 0, cross_fade_samples)
        fade_in = np.linspace(0, 1, cross_fade_samples)

        segments = self.split_text(
            text,
            window_size=window_size,
            split_fn=split_fn,
            length_threshold=length_threshold
        )
        prompts = [process_prompt_control(segment, gender, pitch, speed) for segment in segments]

        completion = ""
        semantic_tokens = []

        audio_index = 0
        last_audio = None

        for i in range(len(segments)):
            prompt = prompts[i]
            if acoustic_tokens is not None:
                prompt += acoustic_tokens.prompt
            if i > 0 and acoustic_tokens is None:
                err_msg = "未成功生成音色prompt，本次推理失败"
                logger.error(err_msg)
                break
            async for tokens in self.generator.async_stream_generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    **kwargs
            ):
                tokens = tokens.text
                if acoustic_tokens is None:
                    completion += tokens
                    acoustics = re.findall(
                        r"(<\|start_acoustic_token\|>.*?<\|end_global_token\|>)",
                        completion)
                    if len(acoustics) > 0:
                        acoustic_tokens = SparkAcousticTokens(acoustics[0])
                        completion = ""
                    else:
                        continue
                else:
                    pred_semantic_tokens = [
                        int(token) for token in re.findall(r"bicodec_semantic_(\d+)", tokens)]

                    semantic_tokens.extend(pred_semantic_tokens)

                if len(semantic_tokens) >= chunk_size:
                    processed = await self._post_process_chunk_audio(
                        audio_idx=audio_index,
                        semantic_tokens=semantic_tokens,
                        chunk_size=chunk_size,
                        global_tokens=acoustic_tokens.global_tokens,
                        cross_fade_samples=cross_fade_samples,
                        fade_in=fade_in,
                        fade_out=fade_out,
                        last_chunk_audio=last_audio,
                        overlap_chunk_size=overlap_chunk_size,
                    )
                    yield processed['yield_audio']

                    audio_index = processed['audio_idx']
                    last_audio = processed['last_chunk_audio']
                    semantic_tokens = processed['semantic_tokens']
                    # increase chunk size for better speech quality
                    chunk_size = min(max_chunk_size, int(chunk_size * audio_chunk_size_scale_factor))

        if len(semantic_tokens) > 0:
            processed = await self._post_process_chunk_audio(
                audio_idx=audio_index,
                semantic_tokens=semantic_tokens,
                chunk_size=len(semantic_tokens),
                global_tokens=acoustic_tokens.global_tokens,
                cross_fade_samples=cross_fade_samples,
                fade_in=fade_in,
                fade_out=fade_out,
                last_chunk_audio=last_audio,
                overlap_chunk_size=overlap_chunk_size,
            )

            yield processed['yield_audio']
            last_audio = processed['last_chunk_audio']

        if last_audio is not None:
            yield last_audio[-cross_fade_samples:]

        if return_acoustic_tokens:
            yield acoustic_tokens

    async def speak_stream_async(
            self,
            text: str,
            name: str = "female",
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
            audio_chunk_duration: float = 1.0,
            max_audio_chunk_duration: float = 8.0,
            audio_chunk_size_scale_factor: float = 2.0,
            audio_chunk_overlap_duration: float = 0.1,
            acoustic_tokens: Optional[SparkAcousticTokens | str] = None,
            return_acoustic_tokens: bool = False,
            **kwargs) -> AsyncIterator[np.ndarray | SparkAcousticTokens]:
        name = name or "female"
        if name not in self.speakers:
            err_msg = f"{name} 角色不存在。"
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.set_seed(seed=self.seed)
        speaker = self.speakers[name]
        out_acoustic_tokens = None
        if name in ['female', 'male']:
            async for chunk in self._control_stream_generate(
                    text=text,
                    gender=name,
                    pitch=pitch,
                    speed=speed,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    length_threshold=length_threshold,
                    window_size=window_size,
                    split_fn=split_fn,
                    audio_chunk_duration=audio_chunk_duration,
                    max_audio_chunk_duration=max_audio_chunk_duration,
                    audio_chunk_size_scale_factor=audio_chunk_size_scale_factor,
                    audio_chunk_overlap_duration=audio_chunk_overlap_duration,
                    acoustic_tokens=acoustic_tokens,
                    return_acoustic_tokens=return_acoustic_tokens,
            ):
                if isinstance(chunk, SparkAcousticTokens):
                    out_acoustic_tokens = chunk
                else:
                    yield (chunk * 32767).astype(np.int16)
        else:
            async for chunk in self._clone_voice_stream_by_tokens(
                    text=text,
                    global_tokens=speaker['global_tokens'],
                    semantic_tokens=speaker['semantic_tokens'],
                    reference_text=speaker['reference_text'],
                    pitch=pitch,
                    speed=speed,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    length_threshold=length_threshold,
                    window_size=window_size,
                    split_fn=split_fn,
                    audio_chunk_duration=audio_chunk_duration,
                    max_audio_chunk_duration=max_audio_chunk_duration,
                    audio_chunk_size_scale_factor=audio_chunk_size_scale_factor,
                    audio_chunk_overlap_duration=audio_chunk_overlap_duration,
                    **kwargs
            ):
                yield (chunk * 32767).astype(np.int16)
        if out_acoustic_tokens is not None:
            yield out_acoustic_tokens

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
        self.set_seed(seed=self.seed)
        tokens = await self._tokenize(
            reference_audio
        )
        audio = await self._clone_voice_by_tokens(
            text=text,
            global_tokens=tokens['global_tokens'],
            semantic_tokens=tokens['semantic_tokens'],
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
            split_fn=split_fn,
            **kwargs
        )
        return (audio * 32767).astype(np.int16)

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
            audio_chunk_duration: float = 1.0,
            max_audio_chunk_duration: float = 8.0,
            audio_chunk_size_scale_factor: float = 2.0,
            audio_chunk_overlap_duration: float = 0.1,
            **kwargs) -> AsyncIterator[np.ndarray]:

        self.set_seed(seed=self.seed)
        tokens = await self._tokenize(
            reference_audio
        )
        async for chunk in self._clone_voice_stream_by_tokens(
                text=text,
                global_tokens=tokens['global_tokens'],
                semantic_tokens=tokens['semantic_tokens'],
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
                split_fn=split_fn,
                audio_chunk_duration=audio_chunk_duration,
                max_audio_chunk_duration=max_audio_chunk_duration,
                audio_chunk_size_scale_factor=audio_chunk_size_scale_factor,
                audio_chunk_overlap_duration=audio_chunk_overlap_duration,
                **kwargs
        ):
            yield (chunk * 32767).astype(np.int16)
