# -*- coding: utf-8 -*-
# Time      :2025/3/29 10:57
# Author    :Hui Huang
from typing import Optional, AsyncIterator, List

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.managers.io_struct import GenerateReqInput

from .base_llm import BaseLLM, GenerationResponse

__all__ = ["SglangGenerator"]


class SglangGenerator(BaseLLM):
    def __init__(self,
                 model_path: str,
                 max_length: int = 32768,
                 gpu_memory_utilization: Optional[float] = None,
                 device: str = "cuda",
                 stop_tokens: Optional[list[str]] = None,
                 stop_token_ids: Optional[List[int]] = None,
                 **kwargs):
        engine_kwargs = dict(
            model_path=model_path,
            context_length=max_length,
            # Logging
            log_level="error",
            log_level_http=None,
            log_requests=False,
            show_time_cost=False,
            mem_fraction_static=gpu_memory_utilization,
            device=device,
            **kwargs
        )
        self.model = Engine(
            **engine_kwargs
        )
        super(SglangGenerator, self).__init__(
            tokenizer=model_path,
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
        obj = GenerateReqInput(
            input_ids=prompt_ids,
            sampling_params={
                "n": 1,
                "max_new_tokens": max_tokens,
                "stop_token_ids": self.stop_token_ids,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "skip_special_tokens": skip_special_tokens,
                **kwargs
            },
            stream=False,
        )
        generator = self.model.tokenizer_manager.generate_request(obj, None)
        ret = await generator.__anext__()

        if isinstance(ret, dict) and "msg" in ret:
            raise ValueError(ret['msg'])
        choices = []

        if isinstance(ret, dict):
            ret = [ret]

        for idx, ret_item in enumerate(ret):
            choices.append(ret_item['text'])

        return GenerationResponse(
            text=choices[0],
            token_ids=self.tokenizer.encode(choices[0]),
        )

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
        obj = GenerateReqInput(
            input_ids=prompt_ids,
            sampling_params={
                "n": 1,
                "max_new_tokens": max_tokens,
                "stop_token_ids": self.stop_token_ids,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "skip_special_tokens": skip_special_tokens,
                **kwargs
            },
            stream=True,
        )
        generator = self.model.tokenizer_manager.generate_request(obj, None)

        previous_texts = ""
        complete_tokens = []
        async for data in generator:
            cur_token_ids = self.tokenizer.encode(data['text'])
            delta_tokens = cur_token_ids[len(complete_tokens):]
            delta_text = data["text"][len(previous_texts):]
            previous_texts = data["text"]
            complete_tokens = cur_token_ids

            yield GenerationResponse(
                text=delta_text,
                token_ids=delta_tokens,
            )
