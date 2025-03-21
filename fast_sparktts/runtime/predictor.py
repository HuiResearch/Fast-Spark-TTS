# -*- coding: utf-8 -*-
# Time      :2025/3/13 20:37
# Author    :Hui Huang
import asyncio
import os
import re
from typing import Optional, Literal, Tuple

import numpy as np
import torch
import soundfile as sf
from ..utils.token_parser import TASK_TOKEN_MAP, GENDER_MAP, LEVELS_MAP
from .audio_tokenizer import AudioTokenizer
from .vocoder import VoCoder
from .logger import get_logger
from scipy.signal import resample

logger = get_logger()


def process_prompt(
        text: str,
        prompt_text: Optional[str] = None,
        global_token_ids: torch.Tensor = None,
        semantic_token_ids: torch.Tensor = None,
) -> Tuple[str, torch.Tensor]:
    """
    Process input for voice cloning.

    Args:
        text: The text input to be converted to speech.
        prompt_text: Transcript of the prompt audio.
        global_token_ids: Global token IDs extracted from reference audio.
        semantic_token_ids: Semantic token IDs extracted from reference audio.

    Returns:
        Tuple containing the formatted input prompt and global token IDs.
    """
    # Convert global tokens to string format
    global_tokens = "".join(
        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
    )

    # Prepare the input tokens for the model
    if prompt_text is not None and len(prompt_text) > 0:
        # Include semantic tokens when prompt text is provided
        semantic_tokens = "".join(
            [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
        )

        inputs = [
            TASK_TOKEN_MAP["tts"],
            "<|start_content|>",
            prompt_text,
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
            "<|start_semantic_token|>",
            semantic_tokens,
        ]
    else:
        # Without prompt text, exclude semantic tokens
        inputs = [
            TASK_TOKEN_MAP["tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
        ]

    # Join all input components into a single string
    inputs = "".join(inputs)
    return inputs, global_token_ids


def process_prompt_control(
        text: str,
        gender: Optional[Literal["female", "male"]] = "female",
        pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
        speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
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


class AsyncFastSparkTTS:
    def __init__(
            self,
            model_path: str,
            max_length: int = 32768,
            gguf_model_file: Optional[str] = None,
            llm_device: Literal["cpu", "cuda", "auto"] | str = "auto",
            audio_device: Literal["cpu", "cuda", "auto"] | str = "auto",
            vocoder_device: Literal["cpu", "cuda", "auto"] | str = "auto",
            engine: Literal["vllm", "llama-cpp", "sglang", "torch"] = "torch",
            wav2vec_attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            llm_attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
            torch_dtype: Literal['float16', "bfloat16", 'float32', 'auto'] = "auto",
            llm_gpu_memory_utilization: Optional[float] = 0.6,
            cache_implementation: Optional[str] = None,
            batch_size: int = 32,
            wait_timeout: float = 0.01,
            **kwargs,
    ):
        """

        Args:
            model_path: 权重路径
            max_length: llm上下文最大长度
            gguf_model_file: llama cpp加载gguf模型文件，不传入则默认路径为 "{model_path}/LLM/model.gguf"
            llm_device: llm使用的device
            audio_device: audio tokenizer使用的device
            vocoder_device: vocoder使用device
            engine: llm 后端类型
            wav2vec_attn_implementation: audio tokenizer中，wav2vec模型使用attn算子
            llm_gpu_memory_utilization: vllm和sglang暂用显存比例，单卡可降低该参数
            batch_size: 音频处理组件单批次处理的最大请求数。
            wait_timeout:
            **kwargs:
        """
        self.engine_type = engine
        self.llm_device = self._auto_detect_device(llm_device)
        self.audio_device = self._auto_detect_device(audio_device)
        self.vocoder_device = self._auto_detect_device(vocoder_device)

        if engine == "vllm":
            from .generator.vllm_generator import VllmGenerator
            self.generator = VllmGenerator(
                model_path=os.path.join(model_path, "LLM"),
                max_length=max_length,
                device=self.llm_device,
                max_num_seqs=batch_size,
                gpu_memory_utilization=llm_gpu_memory_utilization,
                dtype=torch_dtype,
                **kwargs)
        elif engine == "llama-cpp":
            from .generator.llama_cpp_generator import LlamaCPPGenerator
            self.generator = LlamaCPPGenerator(
                model_path=os.path.join(model_path, "LLM"),
                gguf_model_file=gguf_model_file,
                max_length=max_length,
                **kwargs)
        elif engine == 'sglang':
            from .generator.sglang_generator import SglangGenerator
            self.generator = SglangGenerator(
                model_path=os.path.join(model_path, "LLM"),
                max_length=max_length,
                device=self.llm_device,
                gpu_memory_utilization=llm_gpu_memory_utilization,
                max_running_requests=batch_size,
                dtype=torch_dtype,
                **kwargs)
        elif engine == 'torch':
            from .generator.torch_generator import TorchGenerator

            self.generator = TorchGenerator(
                model_path=os.path.join(model_path, "LLM"),
                max_length=max_length,
                device=self.llm_device,
                attn_implementation=llm_attn_implementation,
                torch_dtype=torch_dtype,
                cache_implementation=cache_implementation,
                **kwargs)
        else:

            raise ValueError(f"Unknown backend: {engine}")

        self.audio_tokenizer = AudioTokenizer(
            model_path,
            device=self.audio_device,
            attn_implementation=wav2vec_attn_implementation,
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )
        self.vocoder = VoCoder(
            model_path=os.path.join(model_path, "BiCodec"),
            device=self.vocoder_device,
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )

    def _auto_detect_device(self, device: str):
        if self.engine_type == 'sglang' and re.match("cuda:\d+", device):
            logger.warning(
                "sglang目前不支持指定GPU ID，将默认使用第一个GPU。您可以通过设置环境变量CUDA_VISIBLE_DEVICES=0 来指定GPU。")
            return "cuda"
        if device in ["cpu", "cuda"] or device.startswith("cuda"):
            return device
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    async def async_clone_voice_from_ndarray(
            self,
            text: str,
            reference_wav: np.ndarray,
            reference_wav_len: np.ndarray,
            reference_text: Optional[str] = None,
            temperature: float = 0.6,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 2048,
            **kwargs
    ) -> np.ndarray:
        audio_output = await self.audio_tokenizer.async_process(
            request={
                "reference_wav": reference_wav,
                "reference_wav_len": reference_wav_len
            })
        global_tokens, semantic_tokens = audio_output['global_tokens'], audio_output['semantic_tokens']
        prompt, global_token_ids = process_prompt(
            text=text,
            prompt_text=reference_text,
            global_token_ids=global_tokens,
            semantic_token_ids=semantic_tokens,
        )
        generated_output = await self.generator.async_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )
        pred_semantic_tokens = [int(token) for token in re.findall(r"bicodec_semantic_(\d+)", generated_output)]
        if len(pred_semantic_tokens) == 0:
            logger.error(f"Semantic tokens 预测为空，输入text为：{text}，llm输出为：{generated_output}")
            raise ValueError(f"Semantic tokens 预测为空，输入text为：{text}，llm输出为：{generated_output}")
        pred_semantic_ids = (
            torch.tensor(pred_semantic_tokens)
            .unsqueeze(0).to(torch.int32)
        )
        audio = await self.vocoder.async_process(
            request={
                "global_tokens": global_token_ids,
                "semantic_tokens": pred_semantic_ids,
            }
        )
        audio = audio['waveform'][0].cpu().numpy().astype(np.float32)
        return audio

    @classmethod
    async def prepare_clone_inputs(
            cls,
            text: str,
            reference_audio,
            reference_text: Optional[str] = None,
            temperature: float = 0.6,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 2048,
            **kwargs
    ):
        waveform, sr = sf.read(reference_audio)
        if sr != 16000:
            sample_ratio = 16000 / sr
            waveform = resample(waveform, int(len(waveform) * sample_ratio))
            logger.warning("输入参考音频采样率不为16000，已对其自动进行重采样。")

        lengths = np.array([len(waveform)], dtype=np.int32)
        samples = np.array(waveform, dtype=np.float32)
        samples = samples.reshape(1, -1).astype(np.float32)
        return dict(
            reference_wav=samples,
            reference_wav_len=lengths,
            reference_text=reference_text,
            text=text,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            **kwargs
        )

    async def async_clone_voice(
            self,
            text: str,
            reference_audio: str,
            reference_text: Optional[str] = None,
            temperature: float = 0.6,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 2048,
            **kwargs) -> np.ndarray:

        inputs = await self.prepare_clone_inputs(
            reference_audio=reference_audio,
            reference_text=reference_text,
            text=text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )
        audio = await self.async_clone_voice_from_ndarray(
            **inputs
        )
        return audio

    async def async_generate_voice(
            self,
            text: str,
            gender: Optional[Literal["female", "male"]] = "female",
            pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            temperature: float = 0.6,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 2048,
            **kwargs) -> np.ndarray:
        prompt = process_prompt_control(text, gender, pitch, speed)
        generated_output = await self.generator.async_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )
        pred_semantic_tokens = [int(token) for token in re.findall(r"bicodec_semantic_(\d+)", generated_output)]
        if len(pred_semantic_tokens) == 0:
            logger.error(f"Semantic tokens 预测为空，输入text为：{text}，llm输出为：{generated_output}")
            raise ValueError(f"Semantic tokens 预测为空。 输入text为：{text}，llm输出为：{generated_output}")
        pred_semantic_ids = (
            torch.tensor(pred_semantic_tokens)
            .unsqueeze(0).to(torch.int32)
        )
        global_tokens = [int(token) for token in re.findall(r"bicodec_global_(\d+)", generated_output)]
        if len(global_tokens) == 0:
            logger.error(f"Global tokens 预测为空，输入text为：{text}，llm输出为：{generated_output}")
            raise ValueError(f"Global tokens 预测为空, 输入text为：{text}，llm输出为：{generated_output}")

        global_token_ids = (
            torch.tensor(global_tokens)
            .long()
            .unsqueeze(0)
        )
        audio = await self.vocoder.async_process(
            request={
                "global_tokens": global_token_ids,
                "semantic_tokens": pred_semantic_ids,
            }
        )
        audio = audio['waveform'][0].cpu().numpy().astype(np.float32)
        return audio

    def clone_voice(
            self,
            text: str,
            reference_audio: str,
            reference_text: Optional[str] = None,
            temperature: float = 0.6,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 2048,
            **kwargs) -> np.ndarray:
        return asyncio.run(self.async_clone_voice(
            reference_audio=reference_audio,
            reference_text=reference_text,
            text=text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        ))

    def generate_voice(
            self,
            text: str,
            gender: Optional[Literal["female", "male"]] = "female",
            pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
            temperature: float = 0.6,
            top_k: int = 50,
            top_p: float = 0.95,
            max_tokens: int = 2048,
            **kwargs) -> np.ndarray:
        return asyncio.run(self.async_generate_voice(
            text=text,
            gender=gender,
            pitch=pitch,
            speed=speed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        ))
