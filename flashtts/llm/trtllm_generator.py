# -*- coding: utf-8 -*-
# @Time    : 2025/5/5 11:28
# @Author  : HuangHui
# @File    : trtllm_generator.py
# @Project: FlashTTS
import os.path
from pathlib import Path
from typing import Optional, AsyncIterator
from .base_llm import BaseLLM, GenerationResponse


class TrtLLMGenerator(BaseLLM):
    def __init__(
            self,
            model_path: str,
            tensorrt_path: Optional[str] = None,
            max_length: int = 32768,
            device: str = "cuda",
            stop_tokens: Optional[list[str]] = None,
            stop_token_ids: Optional[list[int]] = None,
            batch_size: int = 4,
            **kwargs):
        from tensorrt_llm import LLM

        assert device == 'cuda'

        if tensorrt_path is None:
            tensorrt_path = os.path.join(model_path, 'tensorrt-engine')

        if not any(Path(tensorrt_path).glob(f'*engine')):
            raise FileNotFoundError(
                f'No tensorrt engine found at {tensorrt_path}. '
                f'Please refer to `https://github.com/NVIDIA/TensorRT-LLM` to convert the LLM weights into a TensorRT engine file and place it in the {tensorrt_path} directory.')

        self.model = LLM(
            model=tensorrt_path,
            tokenizer=model_path,
            dtype='auto',
            max_batch_size=batch_size,
            max_num_tokens=kwargs.pop('max_num_tokens', max_length * batch_size),
            **kwargs,
        )
        super().__init__(
            tokenizer=model_path,
            max_length=max_length,
            stop_tokens=stop_tokens,
            stop_token_ids=stop_token_ids,
        )

    async def _get_trt_generator(
            self,
            prompt_ids: list[int],
            max_tokens: int = 1024,
            temperature: float = 0.9,
            top_p: float = 0.9,
            top_k: int = 50,
            repetition_penalty: float = 1.0,
            skip_special_tokens: bool = True,
            stream: bool = False,
            **kwargs):
        from tensorrt_llm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            stop=self.stop_tokens,
            stop_token_ids=self.stop_token_ids,
            n=1,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            detokenize=True,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )

        generator = self.model.generate_async(
            inputs=prompt_ids,
            sampling_params=sampling_params,
            streaming=stream
        )
        return generator

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
        generator = await self._get_trt_generator(
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=skip_special_tokens,
            stream=False,
            **kwargs
        )
        final_res = None

        async for res in generator:
            final_res = res
        assert final_res is not None
        choices = []
        for output in final_res.outputs:
            choices.append(GenerationResponse(
                text=output.text,
                token_ids=output.token_ids,
            ))
        return choices[0]

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
        generator = await self._get_trt_generator(
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=skip_special_tokens,
            stream=False,
            **kwargs
        )
        previous_texts = ""
        previous_num_tokens = 0
        async for res in generator:
            for output in res.outputs:
                delta_text = output.text[len(previous_texts):]
                previous_texts = output.text

                delta_token_ids = output.token_ids[previous_num_tokens:]
                previous_num_tokens = len(output.token_ids)

                yield GenerationResponse(
                    text=delta_text,
                    token_ids=delta_token_ids
                )

    def shutdown(self):
        self.model._shutdown()
