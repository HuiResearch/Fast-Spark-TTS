# -*- coding: utf-8 -*-
# Time      :2025/4/01 11:00
# Author    :Dylanoy
from typing import Optional, AsyncIterator

from .base_llm import BaseLLM, GenerationResponse

__all__ = ["MlxLmGenerator"]


class MlxLmGenerator(BaseLLM):
    def __init__(
            self,
            model_path: str,
            max_length: int = 32768,
            stop_tokens: Optional[list[str]] = None,
            stop_token_ids: Optional[list[int]] = None,
            **kwargs):
        from mlx_lm import load, stream_generate

        # Load the model and tokenizer
        self.model, tokenizer = load(model_path)

        if stop_tokens:
            [tokenizer.add_eos_token(token_id) for token_id in stop_tokens]

        if stop_token_ids:
            [tokenizer.add_eos_token(str(token_id)) for token_id in stop_token_ids]

        # Store the generate function for later use
        self.stream_generate_fn = stream_generate

        super(MlxLmGenerator, self).__init__(
            tokenizer=tokenizer,
            max_length=max_length,
            stop_tokens=stop_tokens,
            stop_token_ids=stop_token_ids,
        )

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
        completion_tokens = []
        for response in self.stream_generate_fn(
                self.model, self.tokenizer, prompt_ids, max_tokens=max_tokens):
            completion_tokens.append(response.token)

        # Decode the generated tokens into text
        output = self.tokenizer.decode(
            completion_tokens,
            skip_special_tokens=skip_special_tokens
        )
        return GenerationResponse(text=output, token_ids=completion_tokens)

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
        completion_tokens = []
        previous_texts = ""
        for response in self.stream_generate_fn(
                self.model, self.tokenizer, prompt_ids, max_tokens=max_tokens):
            completion_tokens.append(response.token)

            text = self.tokenizer.decode(completion_tokens, skip_special_tokens=skip_special_tokens)

            delta_text = text[len(previous_texts):]
            previous_texts = text

            yield GenerationResponse(
                text=delta_text,
                token_ids=[response.token],
            )
