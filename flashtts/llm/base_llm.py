# -*- coding: utf-8 -*-
# Time      :2025/3/29 10:45
# Author    :Hui Huang
from typing import List, AsyncIterator, Optional
from transformers import AutoTokenizer
import uuid
from dataclasses import dataclass


@dataclass
class GenerationResponse:
    text: str
    token_ids: List[int]


class BaseLLM:

    def __init__(
            self,
            tokenizer,
            max_length: int,
            stop_tokens: Optional[list[str]] = None,
            stop_token_ids: Optional[List[int]] = None
    ):
        self.max_length = max_length
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        if stop_token_ids is None:
            stop_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids = stop_token_ids + [self.tokenizer.eos_token_id]
        if stop_tokens is None:
            stop_tokens = []
        if len(stop_tokens) > 0:
            stop_token_ids = stop_token_ids + self.tokenizer.convert_tokens_to_ids(stop_tokens)

        self.stop_token_ids = stop_token_ids
        self.stop_tokens = self.tokenizer.convert_ids_to_tokens(self.stop_token_ids)

    def valid_max_tokens(self, max_tokens: int) -> int:
        max_tokens = min(self.max_length - 256, max_tokens)
        return max_tokens

    def tokenize(self, text: str | list[int], max_tokens: int) -> List[int]:
        src_len = self.max_length - max_tokens
        src_len = max(src_len, 256)
        if isinstance(text, str):
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
                padding=False
            )
        else:
            tokens = text
        tokens = tokens[-src_len:]
        return tokens

    @classmethod
    async def random_uid(cls):
        return str(uuid.uuid4().hex)

    async def _stream_generate(
            self,
            prompt_ids: list[int],
            max_tokens: int = 1024,
            temperature: float = 0.9,
            top_p: float = 0.9,
            top_k: int = 50,
            repetition_penalty: float = 1.0,
            skip_special_tokens: bool = True,
            **kwargs
    ) -> AsyncIterator[GenerationResponse]:
        yield NotImplementedError

    async def _generate(
            self,
            prompt_ids: list[int],
            max_tokens: int = 1024,
            temperature: float = 0.9,
            top_p: float = 0.9,
            top_k: int = 50,
            repetition_penalty: float = 1.0,
            skip_special_tokens: bool = True,
            **kwargs) -> GenerationResponse:
        raise NotImplementedError

    async def async_generate(
            self,
            prompt: str | list[int],
            max_tokens: int = 1024,
            temperature: float = 0.9,
            top_p: float = 0.9,
            top_k: int = 50,
            repetition_penalty: float = 1.0,
            skip_special_tokens: bool = True,
            **kwargs
    ) -> GenerationResponse:
        max_tokens = self.valid_max_tokens(max_tokens)
        token_ids = self.tokenize(prompt, max_tokens)
        response = await self._generate(
            prompt_ids=token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )
        return response

    async def async_stream_generate(
            self,
            prompt: str | list[int],
            max_tokens: int = 1024,
            temperature: float = 0.9,
            top_p: float = 0.9,
            top_k: int = 50,
            repetition_penalty: float = 1.0,
            skip_special_tokens: bool = True,
            **kwargs) -> AsyncIterator[GenerationResponse]:
        max_tokens = self.valid_max_tokens(max_tokens)
        token_ids = self.tokenize(prompt, max_tokens)
        async for chunk in self._stream_generate(
                prompt_ids=token_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                skip_special_tokens=skip_special_tokens,
                **kwargs
        ):
            yield chunk
