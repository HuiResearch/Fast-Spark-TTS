# -*- coding: utf-8 -*-
# Time      :2025/3/29 10:57
# Author    :Hui Huang
from threading import Thread
from typing import Literal, AsyncIterator, List

import torch

from .base_llm import BaseLLM, GenerationResponse
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer)
import uuid
from typing import TYPE_CHECKING, Optional
from queue import Queue

if TYPE_CHECKING:
    from transformers import AutoTokenizer

__all__ = ["TorchGenerator"]


class ResIteratorStreamer(TextStreamer):

    def __init__(
            self,
            tokenizer: "AutoTokenizer",
            skip_prompt: bool = False,
            timeout: Optional[float] = None,
            **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.res_queue = Queue()
        self.timeout = timeout
        self.num_tokens = 0

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        printable_text = text[self.print_len:]
        self.print_len += len(printable_text)
        printable_tokens = self.token_cache[self.num_tokens:]
        self.num_tokens = len(self.token_cache)

        self.on_finalized_res(GenerationResponse(text=printable_text, token_ids=printable_tokens))

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len:]
            printable_tokens = self.token_cache[self.num_tokens:]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""
            printable_tokens = []

        self.next_tokens_are_prompt = True
        self.on_finalized_res(GenerationResponse(text=printable_text, token_ids=printable_tokens), stream_end=True)

    def on_finalized_res(self, response: GenerationResponse, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.res_queue.put(response, timeout=self.timeout)
        if stream_end:
            self.res_queue.put(None, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.res_queue.get(timeout=self.timeout)
        if value is None:
            raise StopIteration()
        else:
            return value


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
        self.stop = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        for i, input_id in enumerate(input_ids):
            if i >= len(self.stop):
                self.stop.append(False)

            if input_id[-1] in self.stop_token_ids:
                self.stop[i] = True
            if self.stop[i]:
                input_ids[i][-1] = self.stop_token_ids[0]

        if all(self.stop):
            self.stop = []
            return True
        return False


class TorchGenerator(BaseLLM):
    def __init__(self,
                 model_path: str,
                 max_length: int = 32768,
                 device: str = "cpu",
                 attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
                 torch_dtype: Literal['float16', "bfloat16", 'float32', 'auto'] = "auto",
                 cache_implementation: Optional[str] = None,
                 stop_tokens: Optional[list[str]] = None,
                 stop_token_ids: Optional[List[int]] = None,
                 **kwargs):
        self.device = torch.device(device)
        self.cache_implementation = cache_implementation

        runtime_kwargs = dict(
            pretrained_model_name_or_path=model_path,
            torch_dtype=getattr(torch, torch_dtype, "auto"),
            attn_implementation=attn_implementation,
            **kwargs
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            **runtime_kwargs
        )
        self.model.eval().to(self.device)

        self.streamer: dict[str, ResIteratorStreamer] = {}

        super(TorchGenerator, self).__init__(
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
        input_ids = torch.LongTensor([prompt_ids]).to(self.device)
        stop_criteria = StoppingCriteriaList([StopOnTokens(self.stop_token_ids)])
        generated_ids = self.model.generate(
            input_ids,
            generation_config=GenerationConfig(
                max_length=self.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.stop_token_ids[0],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                cache_implementation=self.cache_implementation,
                **kwargs
            ),
            use_cache=True,
            stopping_criteria=stop_criteria,
        )
        prompt_length = input_ids.size(1)
        completion_ids = generated_ids[0][prompt_length:]
        completions_text = self.tokenizer.decode(completion_ids, skip_special_tokens=skip_special_tokens)

        return GenerationResponse(
            text=completions_text,
            token_ids=completion_ids.detach().cpu().numpy().tolist(),
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
        input_ids = torch.LongTensor([prompt_ids]).to(self.device)
        request_id = str(uuid.uuid4().hex)

        # 避免并发请求时，streamer错乱
        self.streamer[request_id] = ResIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=skip_special_tokens)
        cur_streamer = self.streamer[request_id]
        stop_criteria = StoppingCriteriaList([StopOnTokens(self.stop_token_ids)])

        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=cur_streamer,
            generation_config=GenerationConfig(
                max_length=self.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.stop_token_ids[0],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                cache_implementation=self.cache_implementation,
                **kwargs
            ),
            use_cache=True,
            stopping_criteria=stop_criteria)
        Thread(target=self.model.generate, kwargs=generation_kwargs).start()
        for res in cur_streamer:
            yield res

        self.streamer.pop(request_id)
