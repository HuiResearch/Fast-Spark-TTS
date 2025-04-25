# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/25 11:14
# Author  : Hui Huang
import os
from argparse import Namespace, ArgumentParser
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from flashtts import AutoEngine
from flashtts.commands import BaseCLICommand
from flashtts.logger import get_logger
from flashtts.server.base_router import base_router
from flashtts.server.openai_router import openai_router

logger = get_logger()


def serve_command_factory(args: Namespace):
    return ServerCommand(args)


def find_ref_files(role_path: str, suffix: str = '.wav'):
    if not os.path.isdir(role_path):
        return
    for filename in os.listdir(role_path):
        if filename.endswith(suffix):
            return os.path.join(role_path, filename)
    return


async def load_roles(async_engine: AutoEngine, role_dir: Optional[str] = None):
    # 加载已有的角色音频
    if role_dir is not None and os.path.exists(role_dir):
        logger.info(f"Loading roles from: {role_dir}")
        role_list = os.listdir(role_dir)
        exist_roles = []
        for role in role_list:
            if role in exist_roles:
                logger.warning(f"`{role}` already exists")
                continue
            role_path = os.path.join(role_dir, role)

            wav_file = find_ref_files(role_path, suffix='.wav')
            txt_file = find_ref_files(role_path, suffix='.txt')
            npy_file = find_ref_files(role_path, suffix='.npy')
            if wav_file is None:
                continue

            role_text = None
            if async_engine.engine_name == 'mega':
                if npy_file is None:
                    logger.warning("MegaTTS requires a latent_file (.npy) along with the reference audio for cloning.")
                    continue
                else:
                    ref_audio = (wav_file, npy_file)
            else:
                if txt_file is not None:
                    role_text = open(
                        os.path.join(role_dir, role, "reference_text.txt"),
                        "r",
                        encoding='utf8'
                    ).read().strip()
                ref_audio = wav_file

            exist_roles.append(role)
            await async_engine.add_speaker(
                name=role,
                audio=ref_audio,
                reference_text=role_text,
            )
        logger.info(f"Finished loading roles: {', '.join(exist_roles)}")
    else:
        logger.info(f"No roles found in {role_dir}. You can download the character audio through this link.")


async def warmup_engine(async_engine: AutoEngine):
    logger.info("Warming up...")
    if async_engine.engine_name == 'spark':
        await async_engine.speak_async(
            text="测试音频",
            max_tokens=128
        )
    elif async_engine.engine_name == 'orpheus':
        await async_engine.speak_async(
            text="test audio.",
            max_tokens=128
        )
    elif async_engine.engine_name == 'mega':
        await async_engine._engine._generate(text="测试音频", max_tokens=16)
    logger.info("Warmup complete.")


def build_app(args) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 使用解析到的参数初始化全局 TTS 引擎
        engine = AutoEngine(
            model_path=args.model_path,
            snac_path=args.snac_path,
            lang=args.lang,
            max_length=args.max_length,
            llm_device=args.llm_device,
            tokenizer_device=args.tokenizer_device,
            detokenizer_device=args.detokenizer_device,
            backend=args.backend,
            wav2vec_attn_implementation=args.wav2vec_attn_implementation,
            llm_attn_implementation=args.llm_attn_implementation,
            llm_gpu_memory_utilization=args.llm_gpu_memory_utilization,
            torch_dtype=args.torch_dtype,
            batch_size=args.batch_size,
            llm_batch_size=args.llm_batch_size,
            wait_timeout=args.wait_timeout,
            cache_implementation=args.cache_implementation,
            seed=args.seed
        )
        role_dir = None
        if engine.engine_name == 'spark':
            role_dir = args.role_dir or "data/roles"
        elif engine.engine_name == 'mega':
            role_dir = args.role_dir or "data/mega-roles"
        if role_dir is not None:
            await load_roles(engine, role_dir)
        await warmup_engine(engine)
        # 将 engine 保存到 app.state 中，方便路由中使用
        app.state.engine = engine
        yield

    app = FastAPI(lifespan=lifespan)

    app.include_router(base_router)
    app.include_router(openai_router, prefix="/v1")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    if args.api_key is not None:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if request.method == "OPTIONS":
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + args.api_key:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    return app


class ServerCommand(BaseCLICommand):

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        serve_parser = parser.add_parser("serve", help="CLI tool to serve.")

        serve_parser.add_argument("--model_path", type=str, required=True,
                                  help="Path to the TTS model")

        serve_parser.add_argument("--backend", type=str, required=True,
                                  choices=["llama-cpp", "vllm", "sglang", "torch", "mlx-lm"],
                                  help="Backend type, e.g., llama-cpp, vllm, sglang, mlx-lm, or torch")
        serve_parser.add_argument(
            "--lang", type=str, default=None,
            help="Language type for Orpheus TTS model, e.g., mandarin, french, german, korean, hindi, spanish, italian, spanish_italian, english")
        serve_parser.add_argument("--snac_path", type=str, default=None,
                                  help="Path to the SNAC module for OrpheusTTS")
        serve_parser.add_argument("--llm_device", type=str, default="auto",
                                  help="Device for the LLM, e.g., cpu or cuda")
        serve_parser.add_argument("--tokenizer_device", type=str, default="auto",
                                  help="Device for the audio tokenizer")
        serve_parser.add_argument("--detokenizer_device", type=str, default="auto",
                                  help="Device for the audio detokenizer")
        serve_parser.add_argument("--wav2vec_attn_implementation", type=str, default="eager",
                                  choices=["sdpa", "flash_attention_2", "eager"],
                                  help="Attention implementation method for wav2vec")
        serve_parser.add_argument("--llm_attn_implementation", type=str, default="eager",
                                  choices=["sdpa", "flash_attention_2", "eager"],
                                  help="Attention implementation method for the torch generator")
        serve_parser.add_argument("--max_length", type=int, default=32768,
                                  help="Maximum generation length")
        serve_parser.add_argument("--llm_gpu_memory_utilization", type=float, default=0.6,
                                  help="GPU memory utilization ratio for vllm and sglang backends")
        serve_parser.add_argument("--torch_dtype", type=str, default="auto",
                                  choices=['float16', "bfloat16", 'float32', 'auto'],
                                  help="Data type used by the LLM in torch generator")
        serve_parser.add_argument(
            "--cache_implementation", type=str, default=None,
            help='Name of the cache class used in "generate" for faster decoding. Options: static, offloaded_static, sliding_window, hybrid, mamba, quantized.'
        )
        serve_parser.add_argument("--role_dir", type=str, default=None,
                                  help="Directory containing predefined speaker roles")
        serve_parser.add_argument("--api_key", type=str, default=None,
                                  help="API key for request authentication")
        serve_parser.add_argument("--seed", type=int, default=0, help="Random seed")
        serve_parser.add_argument("--batch_size", type=int, default=1,
                                  help="Max number of audio requests processed in a single batch")
        serve_parser.add_argument("--llm_batch_size", type=int, default=256,
                                  help="Max number of LLM requests processed in a single batch")
        serve_parser.add_argument("--wait_timeout", type=float, default=0.01,
                                  help="Timeout for dynamic batching (in seconds)")
        serve_parser.add_argument("--host", type=str, default="0.0.0.0",
                                  help="Host address for the server")
        serve_parser.add_argument("--port", type=int, default=8000,
                                  help="Port number for the server")
        serve_parser.add_argument("--ssl_keyfile", type=str, default=None,
                                  help="Path to the SSL key file")
        serve_parser.add_argument("--ssl_certfile", type=str, default=None,
                                  help="Path to the SSL certificate file")
        serve_parser.set_defaults(func=serve_command_factory)

    def __init__(self, args: Namespace):
        logger.info("Starting FlashTTS service...")
        logger.info(f"Serving args: {args}")

        self._args = args
        self._app = build_app(args)

    def run(self):
        uvicorn.run(
            self._app,
            host=self._args.host,
            port=self._args.port,
            ssl_keyfile=self._args.ssl_keyfile,
            ssl_certfile=self._args.ssl_certfile
        )
