# -*- coding: utf-8 -*-
# Time      :2025/4/26 08:49
# Author    :Hui Huang
import asyncio
import os.path
from argparse import ArgumentParser, Namespace

from flashtts.commands import BaseCLICommand
from flashtts.commands.utils import add_model_parser, add_generate_parser
from flashtts import get_logger, AutoEngine

logger = get_logger()


def infer_command_factory(args: Namespace):
    return InferCommand(args)


class InferCommand(BaseCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        infer_parser = parser.add_parser("infer", help="CLI tool to infer.")
        infer_parser.add_argument(
            "-i",
            "--input",
            type=str,
            help="The text or txt file that needs to be processed."
        )
        infer_parser.add_argument(
            "-o",
            "--output",
            type=str,
            default="output.wav",
            help="The output path of the generated audio. default to `output.wav`"
        )
        infer_parser.add_argument(
            "--name",
            type=str,
            default=None,
            help="Voice role name. If provided, synthesis will be performed using the built-in audio role voice."
        )
        infer_parser.add_argument(
            "--reference_audio",
            type=str,
            default=None,
            help="Reference audio path (.wav). If specified, the reference audio will be used for voice cloning."
        )
        infer_parser.add_argument(
            "--reference_text",
            type=str,
            default=None,
            help="Reference text. The transcribed text content of the reference audio. It is only required for the SparkTTS model."
        )
        infer_parser.add_argument(
            "--latent_file",
            type=str,
            default=None,
            help="The latent file (.npy) of reference audio. It is only required for the MegaTTS3 model."
        )
        add_model_parser(infer_parser)
        add_generate_parser(infer_parser)
        infer_parser.set_defaults(func=infer_command_factory)

    def __init__(self, args: Namespace):
        logger.info("Start up FlashTTS to perform inference.")
        logger.info(f"Inference args: {args}")

        logger.info(f"Loading model from {args.model_path}...")
        self.engine = AutoEngine(
            model_path=args.model_path,
            snac_path=args.snac_path,
            lang=args.lang,
            max_length=args.max_length,
            llm_device=args.llm_device,
            tokenizer_device=args.tokenizer_device,
            detokenizer_device=args.detokenizer_device,
            backend=args.backend,
            llm_tensorrt_path=args.llm_tensorrt_path,
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
        logger.info("Model loaded.")
        self._args = args

    @classmethod
    def read_input(cls, inp: str) -> str:
        if os.path.isfile(inp) and inp.endswith(".txt"):
            with open(inp, "r", encoding='utf-8') as f:
                return f.read()
        else:
            return inp

    def run(self):
        input_str = self.read_input(self._args.input)
        if self._args.reference_audio is not None:
            if not self.engine._SUPPORT_CLONE:
                logger.error("The model does not support voice cloning.")
                return
            reference_audio = self._args.reference_audio
            if self.engine.engine_name == "mega":
                if self._args.latent_file is None:
                    logger.error("The MegaTTS3 model requires the latent_file argument.")
                    return
                else:
                    latent_file = self._args.latent_file
                    reference_audio = (reference_audio, latent_file)
                    self._args.reference_text = None
            ref_text = self.read_input(
                self._args.reference_text) if self._args.reference_text is not None else self._args.reference_text
            audio = asyncio.run(self.engine.clone_voice_async(
                text=input_str,
                reference_audio=reference_audio,
                reference_text=ref_text,
                pitch=self._args.pitch,
                speed=self._args.speed,
                temperature=self._args.temperature,
                top_k=self._args.top_k,
                top_p=self._args.top_p,
                repetition_penalty=self._args.repetition_penalty,
                max_tokens=self._args.max_tokens
            ))
        else:
            audio = asyncio.run(self.engine.speak_async(
                text=input_str,
                name=self._args.name,
                pitch=self._args.pitch,
                speed=self._args.speed,
                temperature=self._args.temperature,
                top_k=self._args.top_k,
                top_p=self._args.top_p,
                repetition_penalty=self._args.repetition_penalty,
                max_tokens=self._args.max_tokens
            ))
        self.engine.write_audio(audio, self._args.output)

        self.engine.shutdown()