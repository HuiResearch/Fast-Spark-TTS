# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/7 16:09
# Author  : Hui Huang
import time
from typing import Literal, Optional, List

from pydantic import BaseModel, Field


# 定义支持多种方式传入参考音频的请求协议
class CloneRequest(BaseModel):
    text: str = Field(
        ...,
        description="The text to generate audio for."
    )
    reference_audio: Optional[str] = Field(
        default=None,
        description=(
            "A reference audio sample used for voice cloning. "
            "This field accepts either a URL pointing to an audio file or base64 encoded audio data."
        )
    )
    reference_text: Optional[str] = Field(
        default=None,
        description="Optional transcript or description corresponding to the reference audio."
    )
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Field(
        default=None,
        description="Specifies the pitch level for the generated audio. Valid options: 'very_low', 'low', 'moderate', 'high', 'very_high'."
    )
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Field(
        default=None,
        description="Specifies the speed level of the audio output. Valid options: 'very_low', 'low', 'moderate', 'high', 'very_high'."
    )
    temperature: float = Field(
        default=0.9,
        description="Controls the randomness of the audio generation process. "
                    "Higher values lead to more variation in the generated audio."
    )
    top_k: int = Field(
        default=50,
        description="Limits the number of highest probability tokens considered during generation."
    )
    top_p: float = Field(
        default=0.95,
        description="Nucleus sampling parameter. Only tokens with a cumulative probability of 'top_p' are considered."
    )
    repetition_penalty: float = Field(
        default=1.0,
        description="Controls the repetition penalty applied to the generated text. "
                    "Higher values penalize repeated words and phrases."
    )
    max_tokens: int = Field(
        default=4096,
        description="The maximum number of tokens that can be generated in the output."
    )
    length_threshold: int = Field(
        default=50,
        description="The text length threshold for segmentation. "
                    "If the input text exceeds this threshold, it will be split into multiple segments for synthesis."
    )
    window_size: int = Field(
        default=50,
        description="Specifies the window size for each segment during text splitting. "
                    "It determines the number of tokens included in each segment."
    )
    stream: bool = Field(
        default=False,
        description="Determines whether the audio output should be streamed in real-time (True) "
                    "or returned after complete generation (False)."
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description=(
            "The format in which to return audio. Supported formats: mp3, opus, aac, flac, wav, pcm. "
            "Note: PCM returns raw 16-bit samples without headers and AAC is not currently supported."
        )
    )


# 定义角色语音合成请求体
class SpeakRequest(BaseModel):
    text: str = Field(
        ...,
        description="The text to generate audio for."
    )
    name: Optional[str] = Field(
        default=None,
        description="The name of the voice character to be used for speech synthesis."
    )
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Field(
        default=None,
        description="Specifies the pitch level for the generated audio. Valid options: 'very_low', 'low', 'moderate', 'high', 'very_high'."
    )
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Field(
        default=None,
        description="Specifies the speed level of the audio output. Valid options: 'very_low', 'low', 'moderate', 'high', 'very_high'."
    )
    temperature: float = Field(
        default=0.9,
        description="Controls the randomness of the speech synthesis. A higher temperature produces more diverse outputs."
    )
    top_k: int = Field(
        default=50,
        description="Limits the sampling to the top 'k' most probable tokens during generation."
    )
    top_p: float = Field(
        default=0.95,
        description="Nucleus sampling threshold: only tokens with a cumulative probability up to 'top_p' are considered."
    )
    repetition_penalty: float = Field(
        default=1.0,
        description="Controls the repetition penalty applied to the generated text. "
                    "Higher values penalize repeated words and phrases."
    )
    max_tokens: int = Field(
        default=4096,
        description="Specifies the maximum number of tokens to generate in the output."
    )
    length_threshold: int = Field(
        default=50,
        description="If the input text exceeds this token length threshold, it will be split into multiple segments for synthesis."
    )
    window_size: int = Field(
        default=50,
        description="Determines the window size for each text segment when performing segmentation on longer texts."
    )
    stream: bool = Field(
        default=False,
        description="Indicates whether the audio output should be streamed in real-time (True) or returned only after complete synthesis (False)."
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description=(
            "The format in which to return audio. Supported formats: mp3, opus, aac, flac, wav, pcm. "
            "Note: PCM returns raw 16-bit samples without headers and AAC is not currently supported."
        )
    )


# 定义多角色语音合成请求体
class MultiSpeakRequest(BaseModel):
    text: str = Field(
        ...,
        description=(
            "The text for multi-role audio synthesis. For sentences intended for a specific role, "
            "please prefix the sentence with `<role:角色名>`. For example, `<role:tara> Hello, how are you?` "
            "indicates that the sentence should be synthesized using the voice of the 'tara' character."
        )
    )
    temperature: float = Field(
        default=0.9,
        description="Controls the randomness of the audio generation. Higher temperatures yield more diverse outputs."
    )
    top_k: int = Field(
        default=50,
        description="Limits the synthesis to the top 'k' highest probability tokens considered during generation."
    )
    top_p: float = Field(
        default=0.95,
        description="Nucleus sampling parameter: only tokens with a cumulative probability up to 'top_p' are considered."
    )
    repetition_penalty: float = Field(
        default=1.0,
        description="Controls the repetition penalty applied to the generated text. "
                    "Higher values penalize repeated words and phrases."
    )
    max_tokens: int = Field(
        default=4096,
        description="Specifies the maximum number of tokens that can be generated in the output."
    )
    length_threshold: int = Field(
        default=50,
        description=(
            "Defines the text length threshold for segmentation. If the input text exceeds this threshold, "
            "it will be split into multiple segments for synthesis."
        )
    )
    window_size: int = Field(
        default=50,
        description="Specifies the window size for each text segment when splitting longer texts."
    )
    stream: bool = Field(
        default=False,
        description="Indicates whether to stream the audio output in real-time (True) or to return it after complete synthesis (False)."
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description=(
            "The format in which to return audio. Supported formats: mp3, opus, aac, flac, wav, pcm. "
            "Note: PCM returns raw 16-bit samples without headers and AAC is not currently supported."
        )
    )


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "FastTTS"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


# copy from https://github.com/remsky/Kokoro-FastAPI/blob/master/api/src/routers/openai_compatible.py
class OpenAISpeechRequest(BaseModel):
    model: str = Field(
        default=None,
        description="The model to use for generation.",
    )
    input: str = Field(..., description="The text to generate audio for")
    voice: str = Field(
        default=None,
        description="The voice to use for generation. Can be a base voice or a combined voice name.",
    )
    pitch: float = Field(
        default=1.0,
        description="Specifies the pitch level for the generated audio. "
    )
    speed: float = Field(
        default=1.0,
        description="Specifies the speed level of the audio output."
    )
    temperature: float = Field(
        default=0.9,
        description="Controls the randomness of the speech synthesis. A higher temperature produces more diverse outputs."
    )
    top_k: int = Field(
        default=50,
        description="Limits the sampling to the top 'k' most probable tokens during generation."
    )
    top_p: float = Field(
        default=0.95,
        description="Nucleus sampling threshold: only tokens with a cumulative probability up to 'top_p' are considered."
    )
    repetition_penalty: float = Field(
        default=1.0,
        description="Controls the repetition penalty applied to the generated text. "
                    "Higher values penalize repeated words and phrases."
    )
    max_tokens: int = Field(
        default=4096,
        description="Specifies the maximum number of tokens to generate in the output."
    )
    length_threshold: int = Field(
        default=50,
        description="If the input text exceeds this token length threshold, it will be split into multiple segments for synthesis."
    )
    window_size: int = Field(
        default=50,
        description="Determines the window size for each text segment when performing segmentation on longer texts."
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, flac, wav, pcm. PCM format returns raw 16-bit samples without headers. AAC is not currently supported.",
    )
    stream: bool = Field(
        default=True,
        description="If true, audio will be streamed as it's generated. Each chunk will be a complete sentence.",
    )
