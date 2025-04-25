# -*- coding: utf-8 -*-
# Time      :2025/3/29 15:55
# Author    :Hui Huang
from typing import List, Literal
import torch
from snac.layers import Decoder
from snac.vq import ResidualVectorQuantize
from ..base_model import SnacBaseModel
from ..batch_processor import AsyncBatchEngine


class SnacDeTokenizerModel(SnacBaseModel):
    def __init__(
            self,
            encoder_dim=64,
            encoder_rates=[3, 3, 7, 7],
            latent_dim=None,
            decoder_dim=1536,
            decoder_rates=[7, 7, 3, 3],
            attn_window_size=32,
            codebook_size=4096,
            codebook_dim=8,
            vq_strides=[8, 4, 2, 1],
            noise=True,
            depthwise=True,
            **kwargs
    ):
        super().__init__()

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            vq_strides=vq_strides,
        )
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            noise,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )

    @torch.no_grad()
    def forward(self, codes: List[torch.Tensor]) -> torch.Tensor:
        z_q = self.quantizer.from_codes(codes)
        audio_hat = self.decoder(z_q)
        return audio_hat.detach()


class SnacDeTokenizer:
    def __init__(
            self,
            model_path: str,
            device: Literal["cpu", "cuda", "mps"] | str = "cpu",
            batch_size: int = 32,
            wait_timeout: float = 0.01):
        self.device = torch.device(device)
        self.model = SnacDeTokenizerModel.from_pretrained(
            model_path,
        ).to(self.device)

        self._batch_processor = AsyncBatchEngine(
            processing_function=self.batch_detokenize_async,
            batch_size=batch_size,
            wait_timeout=wait_timeout
        )

    @torch.no_grad()
    def detokenize(
            self,
            codes: List[torch.Tensor],
    ) -> torch.Tensor:
        output = self.model(
            [code.to(self.device) for code in codes]
        )
        return output

    async def batch_detokenize_async(
            self, requests: list[list[torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        outputs = []
        for codes in requests:
            audio = self.detokenize(codes).detach().cpu()
            outputs.append({"audio": audio})
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return outputs

    async def detokenize_async(self, request: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        output = await self._batch_processor.add_request(
            single_input=request
        )
        return output.get("feature")
