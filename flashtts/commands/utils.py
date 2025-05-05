# -*- coding: utf-8 -*-
# Time      :2025/4/26 09:03
# Author    :Hui Huang


def add_model_parser(arg_parser):
    arg_parser.add_argument("-m", "--model_path", type=str, required=True,
                            help="Path to the TTS model")

    arg_parser.add_argument("-b", "--backend", type=str, required=True,
                            choices=["llama-cpp", "vllm", "sglang", "torch", "mlx-lm", "tensorrt-llm"],
                            help="Backend type, e.g., llama-cpp, vllm, sglang, mlx-lm, torch, or tensorrt-llm")
    arg_parser.add_argument(
        "--lang", type=str, default=None,
        help="Language type for Orpheus TTS model, e.g., mandarin, french, german, korean, hindi, spanish, italian, spanish_italian, english")
    arg_parser.add_argument(
        "--llm_tensorrt_path", type=str, default=None,
        help="The path where the TensorRT-LLM engine is located. "
             "This directory should contain a `config.json` file and a file with the `.engine` extension. "
             "This is only effective when the `backend` is set to `tensorrt-llm`."
    )
    arg_parser.add_argument("--snac_path", type=str, default=None,
                            help="Path to the SNAC module for OrpheusTTS")
    arg_parser.add_argument("--llm_device", type=str, default="auto",
                            help="Device for the LLM, e.g., cpu or cuda")
    arg_parser.add_argument("--tokenizer_device", type=str, default="auto",
                            help="Device for the audio tokenizer")
    arg_parser.add_argument("--detokenizer_device", type=str, default="auto",
                            help="Device for the audio detokenizer")
    arg_parser.add_argument("--wav2vec_attn_implementation", type=str, default="eager",
                            choices=["sdpa", "flash_attention_2", "eager"],
                            help="Attention implementation method for wav2vec")
    arg_parser.add_argument("--llm_attn_implementation", type=str, default="eager",
                            choices=["sdpa", "flash_attention_2", "eager"],
                            help="Attention implementation method for the torch generator")
    arg_parser.add_argument("--max_length", type=int, default=32768,
                            help="Maximum generation length")
    arg_parser.add_argument("--llm_gpu_memory_utilization", type=float, default=0.6,
                            help="GPU memory utilization ratio for vllm and sglang backends")
    arg_parser.add_argument("--torch_dtype", type=str, default="auto",
                            choices=['float16', "bfloat16", 'float32', 'auto'],
                            help="Data type used by the LLM in torch generator")
    arg_parser.add_argument(
        "--cache_implementation", type=str, default=None,
        help='Name of the cache class used in "generate" for faster decoding. Options: static, offloaded_static, sliding_window, hybrid, mamba, quantized.'
    )
    arg_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    arg_parser.add_argument("--batch_size", type=int, default=1,
                            help="Max number of audio requests processed in a single batch")
    arg_parser.add_argument("--llm_batch_size", type=int, default=256,
                            help="Max number of LLM requests processed in a single batch")
    arg_parser.add_argument("--wait_timeout", type=float, default=0.01,
                            help="Timeout for dynamic batching (in seconds)")


def add_generate_parser(arg_parser):
    arg_parser.add_argument(
        "--pitch",
        type=str,
        default=None,
        choices=["very_low", "low", "moderate", "high", "very_high"],
        help="Pitch shift for the generated audio, e.g., very_low, low, moderate, high, very_high"
    )
    arg_parser.add_argument(
        "--speed",
        type=str,
        default=None,
        choices=["very_low", "low", "moderate", "high", "very_high"],
        help="Speed up for the generated audio, e.g., very_low, low, moderate, high, very_high"
    )
    arg_parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Controls the randomness of the audio generation process. "
             "Higher values lead to more variation in the generated audio."
    )
    arg_parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Limits the number of highest probability tokens considered during generation."
    )
    arg_parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter. Only tokens with a cumulative probability of 'top_p' are considered."
    )
    arg_parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Controls the repetition penalty applied to the generated text. "
             "Higher values penalize repeated words and phrases."
    )
    arg_parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="The maximum number of tokens that can be generated in the output."
    )

