# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/3/14 15:36
# Author  : Hui Huang
from typing import Literal

import torch

from fast_tts import AsyncSparkEngine
import asyncio
import time

long_text = ("今日是二零二五年三月十九日，国内外热点事件聚焦于国际局势、经济政策及社会民生领域。"
             "国际局势中，某国领导人围绕地区冲突停火问题展开对话，双方同意停止攻击对方能源设施并推动谈判，但对全面停火提议的落实仍存分歧。"
             "某地区持续军事行动导致数百人伤亡，引发民众抗议，质疑冲突背后的政治动机。另有一方宣称对连续袭击军事目标负责，称此为对前期打击的回应。"
             "欧洲某国通过争议性财政草案，计划放宽债务限制以支持国防与环保项目，引发经济政策讨论。 "
             "国内动态方面，新修订的市场竞争管理条例将于四月二十日施行，重点规范市场秩序。"
             "多部门联合推出机动车排放治理新规，加强对高污染车辆的监管。"
             "社会层面，某地涉及非法集资的大案持续引发关注，受害人数以万计，涉案金额高达数百亿元，暴露出特定领域投资风险。"
             "经济与科技领域，某科技企业公布年度营收突破三千六百五十九亿元，并上调智能汽车交付目标至三十五万台。"
             "另一巨头宣布全面推动人工智能转型，要求各部门绩效与人工智能应用深度绑定，计划年内推出多项相关产品。"
             "充电基础设施建设加速，公共充电桩总量已接近四百万个，同比增长超六成。 "
             "民生政策方面，多地推出新举措：某地限制顺风车单日接单次数以规范运营，另一地启动职工数字技能培训计划，目标三年内覆盖十万女性从业者。"
             "整体来看，今日热点呈现国际博弈复杂化、国内经济科技加速转型、民生政策精准化调整的特点。")

short_text = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。"

generate_config = dict(
    gender="female",
    pitch="moderate",
    speed="moderate",
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    max_tokens=4096,
)


async def run(
        model_path: str,
        backend: Literal["vllm", "llama-cpp", "sglang", "torch", "mlx-lm"] = "torch",
        device: Literal["cpu", "cuda", "auto"] | str = "auto"
):
    model_kwargs = {
        "model_path": model_path,
        "max_length": 32768,
        "llm_device": device,
        "tokenizer_device": device,
        "detokenizer_device": device,
        "backend": backend
    }

    if torch.cuda.is_available():
        model_kwargs["wav2vec_attn_implementation"] = "sdpa"
        model_kwargs["llm_gpu_memory_utilization"] = 0.6

    if backend == "torch":
        model_kwargs["torch_dtype"] = "bfloat16"
        model_kwargs["llm_attn_implementation"] = 'sdpa'

    model = AsyncSparkEngine(**model_kwargs)

    # warmup
    await model.generate_voice_async(text="你好。", max_tokens=64)

    start_time = time.perf_counter()

    short_audio = await model.generate_voice_async(text=short_text, **generate_config)

    end_time = time.perf_counter()

    long_audio = await model.generate_voice_async(
        text=long_text, **generate_config, length_threshold=50, window_size=50)

    end_time2 = time.perf_counter()

    short_len = len(short_audio) / 16000
    long_len = len(long_audio) / 16000

    print(
        f"短文本推理耗时：{end_time - start_time} s, 短文本输出音频长度：{short_len}，RTF: {(end_time - start_time) / short_len}")
    print(
        f"长文本推理耗时：{end_time2 - end_time} s, 长文本输出音频长度：{long_len}，RTF: {(end_time2 - end_time) / long_len}")


if __name__ == '__main__':
    asyncio.run(run(backend="sglang", model_path="Spark-TTS-0.5B", device="cuda"))
