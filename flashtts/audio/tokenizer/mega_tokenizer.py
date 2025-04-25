# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/16 14:40
# Author  : Hui Huang
import os
import random
from copy import deepcopy

import whisper
import librosa
import pyloudnorm as pyln
import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal
from ..base_model import MegaBaseModel
from ...modules.mega_modules.ar_dur.ar_dur_predictor import CodePredictor, expand_states
from ...modules.mega_modules.ar_dur.layers import LengthRegulator, PosEmb, Embedding
from ...modules.mega_modules.ar_dur.rel_transformer import RelTransformerEncoder
from ...modules.mega_modules.ar_dur.rot_transformer import RotTransformerDecoderLayer
from ...modules.mega_modules.llm_dit.cfm import ConditionalFlowMatcher
from ...modules.mega_modules.llm_dit.time_embedding import TimestepEmbedding
from ...modules.mega_modules.llm_dit.transformer import Transformer
from ...modules.mega_modules.ph_tone_convert import split_ph_timestamp
from ...modules.mega_modules.utils import mel2token_to_dur
from ...modules.mega_modules.wavvae.decoder.diag_gaussian import DiagonalGaussianDistribution
from ...modules.mega_modules.wavvae.decoder.hifigan_modules import Upsample, Generator
from ...modules.mega_modules.wavvae.decoder.seanet_encoder import Encoder
from ...modules.mega_modules.whisper_small import AudioEncoder, TextDecoder


class Whisper(MegaBaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_vocab = 6800
        self.n_text_layer = 6
        self.n_text_head = 8
        self.n_text_ctx = 2048

        self.encoder = AudioEncoder(
            n_mels=80, n_ctx=3000, n_state=512, n_head=8, n_layer=6,
        )
        self.decoder = TextDecoder(
            n_vocab=6800, n_ctx=2048, n_state=512, n_head=8, n_layer=6,
        )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel, None)

    def logits(self, tokens, audio_features, kv_cache=None):
        return self.decoder(tokens, None, audio_features, kv_cache=kv_cache)

    def forward(
            self, mel, mel_len, token, token_len
    ) -> Dict[str, torch.Tensor]:
        attn_mask_enc = self.sequence_mask(mel_len // 2, device=mel.device) > 0
        attn_mask_dec = self.sequence_mask(token_len, device=mel.device) > 0
        return self.decoder(token, attn_mask_dec, self.encoder(mel, attn_mask_enc))

    def sequence_mask(self, seq_lens, max_len=None, device='cpu'):
        if max_len is None:
            max_len = seq_lens.max()
        mask = torch.arange(max_len).unsqueeze(0).to(device)  # [1, t]
        mask = mask < (seq_lens.unsqueeze(1))  # [1, t] + [b, 1] = [b, t]
        mask = mask.float()
        return mask


class WavVAE(MegaBaseModel):
    CKPT_NAME = "model_gen"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(dowmsamples=[6, 5, 4, 4, 2])
        self.proj_to_z = nn.Linear(512, 64)
        self.proj_to_decoder = nn.Linear(32, 320)

        self.latent_upsampler = Upsample(320, 4)
        self.decoder = Generator(
            input_size_=160,
            ngf=128,
            frame_shift=self.config["melgan_config"]["frame_shift"],
            num_band=1,
            use_tanh=self.config["melgan_config"]["use_tanh"],
            ratios=[5, 4, 4, 3])

    def encode_latent(self, audio):
        posterior = self.encode(audio)
        latent = posterior.sample().permute(0, 2, 1)  # (b,t,latent_channel)
        return latent

    def encode(self, audio):
        x = self.encoder(audio).permute(0, 2, 1)
        x = self.proj_to_z(x).permute(0, 2, 1)
        poseterior = DiagonalGaussianDistribution(x)
        return poseterior

    def decode(self, latent):
        latent = self.proj_to_decoder(latent).permute(0, 2, 1)
        return self.decoder(self.latent_upsampler(latent))

    def forward(self, audio):
        posterior = self.encode(audio)
        latent = posterior.sample().permute(0, 2, 1)  # (b, t, latent_channel)
        recon_wav = self.decode(latent)
        return recon_wav, posterior


class ARDurPredictor(MegaBaseModel, CodePredictor):
    CKPT_NAME = "dur_model"

    def __init__(self, config):
        CodePredictor.__init__(
            self,
            config,
            config['dur_txt_hs'],
            config['dur_model_hidden_size'],
            174,
            code_size=config['dur_code_size']
        )
        self.config = config
        bias = config.get('lm_bias', True)
        self.use_rot_embed = self.config.get("use_rot_embed", False)
        if self.use_rot_embed:
            self.layers = nn.ModuleList([])
            self.layers.extend([
                RotTransformerDecoderLayer(
                    self.config['dur_model_hidden_size'],
                    0.0, kernel_size=1,
                    ffn_hidden_size=self.config['dur_model_hidden_size'] * 4,
                    post_ln=self.use_post_ln, op_version=1, bias=bias)
                for _ in range(self.config['dur_model_layers'])
            ])
        if self.config['dur_model_type'] == 'ar_mse':
            self.project_out_dim = nn.Sequential(torch.nn.Linear(self.config['dur_model_hidden_size'], 1),
                                                 nn.Softplus())
        else:
            self.project_out_dim = torch.nn.Linear(
                self.config['dur_model_hidden_size'],
                self.config['dur_code_size'] + 1
            )

    def forward(self, txt_tokens, ling_feas, char_tokens, ph2char, bert_embed,
                prev_code, spk_id=None, spk_embed=None, mels_timbre=None, mel2ph=None,
                incremental_state=None, x_ling=None, attn_mask=None, spk_pos_ids_flat=None,
                prompt_length=None, cache_size=20, streaming=False):
        x = self.code_emb(prev_code)
        if x_ling is None:
            x_ling = self.forward_ling_encoder(
                txt_tokens, ling_feas, char_tokens, ph2char, bert_embed, spk_id, spk_embed, mels_timbre)
            x_ling = x_ling.flatten(0, 1)
            txt_tokens = txt_tokens.flatten(0, 1)
            x_ling = x_ling[txt_tokens > 0][None]

        # run decoder
        self_attn_padding_mask = None
        if self.use_pos_embed:
            positions = self.embed_positions(
                prev_code,
                incremental_state=incremental_state
            )
        if incremental_state is not None:
            x_ling = x_ling[:, x.shape[1] - 1:x.shape[1]]
            if spk_pos_ids_flat is not None:
                spk_pos_ids_flat = spk_pos_ids_flat[:, x.shape[1] - 1:x.shape[1]]
            x = x[:, -1:]
            if self.use_pos_embed:
                positions = positions[:, -1:]
            if streaming:
                # Shift Pos: query pos is min(cache_size, idx)
                spk_pos_ids_flat = torch.min(torch.LongTensor([prompt_length + cache_size]).to(x.device),
                                             spk_pos_ids_flat)

        # # B x T x C -> T x B x C
        if self.use_pos_embed:
            x = x + positions
        x_ling = x_ling[:, :self.config['max_tokens']].contiguous()
        T = min(self.config.get('max_tokens_per_item', 1e9), x_ling.shape[1])
        x_ling = x_ling.reshape(-1, T, x_ling.shape[-1])
        x = x + x_ling
        x = x.transpose(0, 1)

        for idx, layer in enumerate(self.layers):
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
                if attn_mask is not None:
                    self_attn_mask = self_attn_mask + (1 - attn_mask.float()) * -1e8
                self_attn_mask = self_attn_mask.clamp_min(-1e8)
            else:
                self_attn_mask = None

            x, attn_weights = layer(
                x,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                spk_pos_ids_flat=spk_pos_ids_flat
            )

        if streaming and incremental_state != {}:
            for k, v in incremental_state.items():
                if 'attn_state' in k:
                    prev_key, prev_value = incremental_state[k]['prev_key'], incremental_state[k]['prev_value']
                    cur_length = prev_key.shape[2]
                    if cur_length - prompt_length > cache_size:
                        prev_key = torch.cat((prev_key[:, :, :prompt_length], prev_key[:, :, -cache_size:]), dim=2)
                        prev_value = torch.cat((prev_value[:, :, :prompt_length], prev_value[:, :, -cache_size:]),
                                               dim=2)
                    incremental_state[k]['prev_key'], incremental_state[k]['prev_value'] = prev_key, prev_value

        if not self.use_post_ln:
            x = self.layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = self.project_out_dim(x)
        return x

    def infer(self, txt_tokens, ling_feas, char_tokens, ph2char, bert_embed,
              spk_id=None, spk_embed=None, mels_timbre=None,
              incremental_state=None, ctx_vqcodes=None, spk_pos_ids_flat=None, return_state=False,
              first_step_min=0, return_probs=False, first_decoder_inp=None, dur_disturb=0.0, **kwargs):
        if incremental_state is None:
            incremental_state = {}
        x_ling = self.forward_ling_encoder(
            txt_tokens, ling_feas, char_tokens, ph2char, bert_embed,
            spk_id, spk_embed, mels_timbre)
        x_ling = x_ling.flatten(0, 1)
        txt_tokens_ori = txt_tokens
        txt_tokens_withpad = txt_tokens = txt_tokens.flatten(0, 1)
        x_ling = x_ling[txt_tokens > 0][None]
        txt_tokens = txt_tokens[txt_tokens > 0][None]

        decoded = torch.zeros_like(txt_tokens)
        decoded = F.pad(decoded, [1, 0], value=self.code_size + 1)
        if incremental_state != {}:
            if first_decoder_inp is None:
                assert ctx_vqcodes is not None
                decoded[:, :ctx_vqcodes.shape[1]] = ctx_vqcodes
                ctx_vqcodes = None
            else:
                decoded[:, :1] = first_decoder_inp
        probs = []
        for step in range(decoded.shape[1] - 1):
            vq_pred = self(txt_tokens, None, None, None, None,
                           decoded[:, :step + 1], None, None, None,
                           incremental_state=incremental_state, x_ling=x_ling,
                           spk_pos_ids_flat=spk_pos_ids_flat, **kwargs)
            probs.append(vq_pred.cpu())
            if ctx_vqcodes is None or step >= ctx_vqcodes.shape[1]:
                if self.config['dur_model_type'] == 'ar_mse':
                    d = vq_pred[:, -1, 0]
                    if dur_disturb > 0 and step >= 1:
                        if random.random() > 0.5:
                            d = d * (1 + random.random() * dur_disturb)
                        else:
                            d = d / (1 + random.random() * dur_disturb)
                        d = torch.clamp_max(d, self.code_size - 1)
                    vq_pred = torch.round(d).long()
                else:
                    vq_pred = self.sample_one_step(vq_pred)
                decoded[:, step + 1] = torch.clamp_min(vq_pred, 1)
                if step == 0:
                    decoded[:, step + 1] = torch.clamp_min(vq_pred, first_step_min)
            else:
                decoded[:, step + 1] = ctx_vqcodes[:, step]
        decoded = decoded[:, 1:]
        decoded_2d = torch.zeros_like(txt_tokens_ori)
        decoded_2d.flatten(0, 1)[txt_tokens_withpad > 0] = decoded
        if return_state:
            return decoded_2d, incremental_state
        if return_probs:
            return decoded_2d, torch.cat(probs, 1)
        return decoded_2d

    def streaming_infer(self, txt_tokens, ling_feas, char_tokens, ph2char, bert_embed,
                        spk_id=None, spk_embed=None, mels_timbre=None,
                        incremental_state=None, ctx_vqcodes=None, spk_pos_ids_flat=None, return_state=False,
                        **kwargs):
        if incremental_state is None:
            incremental_state = {}
        x_ling = self.forward_ling_encoder(
            txt_tokens, ling_feas, char_tokens, ph2char, bert_embed,
            spk_id, spk_embed, mels_timbre)
        x_ling = x_ling.flatten(0, 1)
        txt_tokens_ori = txt_tokens
        txt_tokens_withpad = txt_tokens = txt_tokens.flatten(0, 1)
        x_ling = x_ling[txt_tokens > 0][None]
        txt_tokens = txt_tokens[txt_tokens > 0][None]

        vq_decoded = torch.zeros_like(txt_tokens)
        vq_decoded = F.pad(vq_decoded, [1, 0], value=self.code_size + 1)
        if incremental_state != {}:
            assert ctx_vqcodes is not None
            vq_decoded[:, :ctx_vqcodes.shape[1]] = ctx_vqcodes
            ctx_vqcodes = None
        prompt_length = list(incremental_state.items())[0][1]['prev_key'].shape[2]
        for step in range(vq_decoded.shape[1] - 1):
            vq_pred = self(txt_tokens, None, None, None, None,
                           vq_decoded[:, :step + 1], None, None, None,
                           incremental_state=incremental_state, x_ling=x_ling,
                           spk_pos_ids_flat=spk_pos_ids_flat, prompt_length=prompt_length, streaming=True, **kwargs)
            if ctx_vqcodes is None or step >= ctx_vqcodes.shape[1]:
                if self.config['dur_model_type'] == 'ar_mse':
                    vq_pred = torch.round(vq_pred[:, -1, 0]).long()
                else:
                    vq_pred = self.sample_one_step(vq_pred)
                vq_decoded[:, step + 1] = vq_pred
            else:
                vq_decoded[:, step + 1] = ctx_vqcodes[:, step]
        vq_decoded = vq_decoded[:, 1:]
        vq_decoded_2d = torch.zeros_like(txt_tokens_ori)
        vq_decoded_2d.flatten(0, 1)[txt_tokens_withpad > 0] = vq_decoded
        if return_state:
            return vq_decoded_2d, incremental_state
        return vq_decoded_2d


class Diffusion(MegaBaseModel):
    CKPT_NAME = "dit"

    def __init__(self, config=None):
        super().__init__()
        # Hparams
        # cond dim
        self.local_cond_dim = 512
        self.ctx_mask_dim = 16
        self.in_channels = 32
        self.out_channels = 32
        # LLM
        self.encoder_dim = 1024
        self.encoder_n_layers = 24
        self.encoder_n_heads = 16
        self.max_seq_len = 16384
        self.multiple_of = 256

        self.ctx_mask_proj = nn.Linear(1, self.ctx_mask_dim)
        self.local_cond_project = nn.Linear(
            self.out_channels + self.ctx_mask_dim, self.local_cond_dim)

        self.encoder = Transformer(self.encoder_n_layers, self.encoder_dim, self.encoder_n_heads, self.max_seq_len)

        self.x_prenet = nn.Linear(self.in_channels, self.encoder_dim)
        self.prenet = nn.Linear(self.local_cond_dim, self.encoder_dim)
        self.postnet = nn.Linear(self.encoder_dim, self.out_channels)

        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        # The implementation of TimestepEmbedding is a modified version from F5-TTS (https://github.com/SWivid/F5-TTS),
        # which is licensed under the MIT License.
        self.f5_time_embed = TimestepEmbedding(self.encoder_dim)

        # text encoder
        self.ph_encoder = RelTransformerEncoder(
            302, self.encoder_dim, self.encoder_dim,
            self.encoder_dim * 2, 4, 6,
            3, 0.0, prenet=True, pre_ln=True)
        self.tone_embed = Embedding(32, self.encoder_dim, padding_idx=0)
        self.ph_pos_embed = PosEmb(self.encoder_dim)
        self.ling_pre_net = torch.nn.Sequential(*[
            torch.nn.Conv1d(self.encoder_dim, self.encoder_dim, kernel_size=s * 2, stride=s, padding=s // 2)
            for i, s in enumerate([2, 2])
        ])

    def forward(self, inputs):
        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask

        """ local conditioning (prompt_latent + spk_embed) """
        ctx_mask_emb = self.ctx_mask_proj(ctx_mask)
        # ctx_feature = ctx_feature * (1 - inputs["spk_cfg_mask"][:, :, None])
        local_cond = torch.cat([ctx_feature, ctx_mask_emb], dim=-1)
        local_cond = self.local_cond_project(local_cond)

        """ diffusion target latent """
        x = inputs['lat']

        # Here, x is x1 in CFM
        x0 = torch.randn_like(x)
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x)

        # define noisy_input and target
        t = t.bfloat16()
        x_noisy = (xt * (1 - ctx_mask)).bfloat16()
        target = ut

        # concat condition.
        x_ling = self.forward_ling_encoder(inputs["phone"], inputs["tone"])
        x_ling = self.ling_pre_net(expand_states(x_ling, inputs['mel2ph']).transpose(1, 2)).transpose(1, 2)
        x_noisy = self.x_prenet(x_noisy) + self.prenet(local_cond) + x_ling
        encoder_out = self.encoder(x_noisy, self.f5_time_embed(t), attn_mask=inputs["text_mel_mask"],
                                   do_checkpoint=False)
        pred = self.postnet(encoder_out)

        return pred, target

    def forward_ling_encoder(self, txt_tokens, tone_tokens):
        ph_tokens = txt_tokens
        ph_nonpadding = (ph_tokens > 0).float()[:, :, None]  # [B, T_phone, 1]

        # enc_ph
        ph_enc_oembed = self.tone_embed(tone_tokens)
        ph_enc_oembed = ph_enc_oembed + self.ph_pos_embed(
            torch.arange(0, ph_tokens.shape[1])[None,].to(ph_tokens.device))
        ph_enc_oembed = ph_enc_oembed
        ph_enc_oembed = ph_enc_oembed * ph_nonpadding
        x_ling = self.ph_encoder(ph_tokens, other_embeds=ph_enc_oembed) * ph_nonpadding
        return x_ling

    def _forward(self, x, local_cond, x_ling, timesteps, ctx_mask, dur=None, seq_cfg_w=[1.0, 1.0]):
        """ When we use torchdiffeq, we need to include the CFG process inside _forward() """
        x = x * (1 - ctx_mask)
        x = self.x_prenet(x) + self.prenet(local_cond) + x_ling
        pred_v = self.encoder(x, self.f5_time_embed(timesteps),
                              attn_mask=torch.ones((x.size(0), x.size(1)), device=x.device))
        pred = self.postnet(pred_v)

        """ Perform multi-cond CFG """
        cond_spk_txt, cond_txt, uncond = pred.chunk(3)
        pred = uncond + seq_cfg_w[0] * (cond_txt - uncond) + seq_cfg_w[1] * (cond_spk_txt - cond_txt)
        return pred

    @torch.no_grad()
    def inference(self, inputs, timesteps=20, seq_cfg_w=[1.0, 1.0], **kwargs):
        # txt embedding
        x_ling = self.forward_ling_encoder(inputs["phone"], inputs["tone"])
        x_ling = self.ling_pre_net(expand_states(x_ling, inputs['dur']).transpose(1, 2)).transpose(1, 2)

        # speaker embedding
        ctx_feature = inputs['lat_ctx']
        ctx_feature[1:, :, :] = 0  # prefix spk cfg
        ctx_mask_emb = self.ctx_mask_proj(inputs['ctx_mask'])

        # local conditioning.
        local_cond = torch.cat([ctx_feature, ctx_mask_emb], dim=-1)
        local_cond = self.local_cond_project(local_cond)

        ''' Euler ODE solver '''
        bsz, device, frm_len = (local_cond.size(0), local_cond.device, local_cond.size(1))
        # Sway sampling from F5-TTS (https://github.com/SWivid/F5-TTS),
        # which is licensed under the MIT License.
        sway_sampling_coef = -1.0
        t_schedule = torch.linspace(0, 1, timesteps + 1, device=device, dtype=x_ling.dtype)
        if sway_sampling_coef is not None:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)

        # AMO sampling implementation for "AMO Sampler: Enhancing Text Rendering with Overshooting" (https://arxiv.org/pdf/2411.19415)
        def amo_sampling(z_t, t, t_next, v):
            # Upcast to avoid precision issues when computing prev_sample
            z_t = z_t.to(torch.float32)

            # Constant definition in Algorithm 1
            s = t_next
            c = 3

            # Line 7 in Algorithm 1
            o = min(t_next + c * (t_next - t), 1)
            pred_z_o = z_t + (o - t) * v

            # Line 11 in Algorithm 1
            a = s / o
            b = ((1 - s) ** 2 - (a * (1 - o)) ** 2) ** 0.5
            noise_i = torch.randn(size=z_t.shape, device=z_t.device)
            z_t_next = a * pred_z_o + b * noise_i
            return z_t_next.to(v.dtype)

        x = torch.randn([1, frm_len, self.out_channels], device=device)
        for step_index in range(timesteps):
            x = x.to(torch.float32)
            sigma = t_schedule[step_index].to(x_ling.dtype)
            sigma_next = t_schedule[step_index + 1]
            model_out = self._forward(torch.cat([x] * bsz), local_cond, x_ling, timesteps=sigma.unsqueeze(0),
                                      ctx_mask=inputs['ctx_mask'], dur=inputs['dur'], seq_cfg_w=seq_cfg_w)
            x = amo_sampling(x, sigma, sigma_next, model_out)
            # Cast sample back to model compatible dtype
            x = x.to(model_out.dtype)

        return x


class MegaTokenizer:
    def __init__(
            self,
            model_path: str,
            device: Literal["cpu", "cuda", "mps"] | str = "cpu"
    ):
        self.device_type = device
        self.device = torch.device(device)
        self.whisper = Whisper.from_pretrained(
            os.path.join(model_path, "aligner_lm"))
        self.whisper.to(self.device)

        self.wavvae = WavVAE.from_pretrained(
            os.path.join(model_path, "wavvae")
        )
        self.wavvae.to(self.device)

        self.dur_model = ARDurPredictor.from_pretrained(
            os.path.join(model_path, "duration_lm")
        )
        self.dur_model.to(self.device)

        self.dit = Diffusion.from_pretrained(
            os.path.join(model_path, "diffusion_transformer")
        )
        self.dit.to(self.device)

        self.sample_rate = 24000
        self.fm = 8
        self.loudness_meter = pyln.Meter(self.sample_rate)
        self.length_regulator = LengthRegulator()
        self.dtype = getattr(torch, self._get_dtype(device))
        self.cfg_mask_token_phone = 302 - 1
        self.cfg_mask_token_tone = 32 - 1
        self.vae_stride = self.wavvae.config.get('vae_stride', 4)
        self.hop_size = self.wavvae.config.get('hop_size', 4)

    def _get_dtype(self, device):
        if device.startswith('cuda'):
            # bf16数值转换暂时有问题
            dtype = "float16"
        else:
            dtype = 'float32'
        return dtype

    def load_audio(self, audio):
        wav, sr = sf.read(audio)
        if len(wav.shape) > 1:
            wav = wav[:, 0]

        if sr != self.sample_rate:
            wav = soxr.resample(wav, sr, self.sample_rate, quality="VHQ")
        ws = self.whisper.config['win_size']
        if len(wav) % ws < ws - 1:
            wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0).astype(np.float32)
        wav = np.pad(wav, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)
        loudness_prompt = self.loudness_meter.integrated_loudness(wav.astype(float))
        return {
            "wav": wav,
            "loudness_prompt": loudness_prompt
        }

    @torch.no_grad()
    def align(self, wav):
        whisper_wav = librosa.resample(wav, orig_sr=self.sample_rate, target_sr=16000)
        mel = torch.FloatTensor(whisper.log_mel_spectrogram(whisper_wav).T).to(self.device)[None].transpose(1, 2)
        prompt_max_frame = mel.size(2) // self.fm * self.fm
        mel = mel[:, :, :prompt_max_frame]
        token = torch.LongTensor([[798]]).to(self.device)
        audio_features = self.whisper.embed_audio(mel)
        for i in range(768):
            with torch.amp.autocast(self.device_type, dtype=self.dtype):
                logits = self.whisper.logits(token, audio_features, None)
                token_pred = torch.argmax(F.softmax(logits[:, -1], dim=-1), 1)[None]
                token = torch.cat([token, token_pred], dim=1)
                if token_pred[0] == 799:
                    break
        alignment_tokens = token

        ph_ref, tone_ref, dur_ref, _ = split_ph_timestamp(deepcopy(alignment_tokens)[0, 1:-1])
        ph_ref = torch.Tensor(ph_ref)[None].to(self.device)
        tone_ref = torch.Tensor(tone_ref)[None].to(self.device)
        if dur_ref.sum() < prompt_max_frame:
            dur_ref[-1] += prompt_max_frame - dur_ref.sum()
        elif dur_ref.sum() > prompt_max_frame:
            len_diff = dur_ref.sum() - prompt_max_frame
            while True:
                for i in range(len(dur_ref)):
                    dur_ref[i] -= 1
                    len_diff -= 1
                    if len_diff == 0:
                        break
                if len_diff == 0:
                    break
        mel2ph_ref = self.length_regulator(dur_ref[None]).to(self.device)
        mel2ph_ref = mel2ph_ref[:, :mel2ph_ref.size(1) // self.fm * self.fm]
        return {
            "ph_ref": ph_ref.detach().cpu(),
            "tone_ref": tone_ref.detach().cpu(),
            "mel2ph_ref": mel2ph_ref.detach().cpu(),
        }

    @torch.no_grad()
    def make_dur_prompt(self, ref_dict: dict):
        dur_tokens_2d_ = mel2token_to_dur(
            ref_dict['mel2ph_ref'], ref_dict["ph_ref"].shape[1]).clamp(
            max=self.dur_model.config['dur_code_size'] - 1) + 1

        ctx_dur_tokens = dur_tokens_2d_.clone().flatten(0, 1).to(self.device)
        txt_tokens_flat_ = ref_dict["ph_ref"].flatten(0, 1)
        ctx_dur_tokens = ctx_dur_tokens[txt_tokens_flat_ > 0][None]

        last_dur_pos_prompt = ctx_dur_tokens.shape[1]
        dur_spk_pos_ids_flat = range(0, last_dur_pos_prompt)
        dur_spk_pos_ids_flat = torch.LongTensor([dur_spk_pos_ids_flat]).to(self.device)
        with torch.amp.autocast(self.device_type, dtype=self.dtype):
            _, incremental_state_dur_prompt = self.dur_model.infer(
                ref_dict["ph_ref"].to(self.device),
                {'tone': ref_dict["tone_ref"].to(self.device)},
                None, None, None,
                ctx_vqcodes=ctx_dur_tokens, spk_pos_ids_flat=dur_spk_pos_ids_flat, return_state=True)
        return {
            'incremental_state_dur_prompt': incremental_state_dur_prompt,
            'ctx_dur_tokens': ctx_dur_tokens.detach().cpu(),
        }

    @torch.no_grad()
    def preprocess(self, audio, latent_file, topk_dur=1):
        wav_dict = self.load_audio(audio)
        ref_dict = self.align(wav_dict["wav"])

        vae_latent = torch.from_numpy(np.load(latent_file)).to(self.device)
        vae_latent = vae_latent[:, :ref_dict['mel2ph_ref'].size(1) // 4]

        self.dur_model.config["infer_top_k"] = topk_dur if topk_dur > 1 else None
        dur_prompt = self.make_dur_prompt(ref_dict)
        return {
            'ph_ref': ref_dict['ph_ref'],
            'tone_ref': ref_dict['tone_ref'],
            'mel2ph_ref': ref_dict['mel2ph_ref'],
            'vae_latent': vae_latent.detach().cpu(),
            'incremental_state_dur_prompt': dur_prompt['incremental_state_dur_prompt'],
            'ctx_dur_tokens': dur_prompt['ctx_dur_tokens'],
            "loudness_prompt": wav_dict["loudness_prompt"],
        }

    @torch.no_grad()
    def dur_predict(
            self,
            ctx_dur_tokens,
            incremental_state_dur_prompt,
            ph_pred,
            tone_pred,
            dur_disturb,
            dur_alpha,
            is_first,
            is_final):
        last_dur_token = ctx_dur_tokens[:, -1:]
        last_dur_pos_prompt = ctx_dur_tokens.shape[1]
        incremental_state_dur = deepcopy(incremental_state_dur_prompt)
        txt_len = ph_pred.shape[1]
        dur_spk_pos_ids_flat = range(last_dur_pos_prompt, last_dur_pos_prompt + txt_len)
        dur_spk_pos_ids_flat = torch.LongTensor([dur_spk_pos_ids_flat]).to(self.device)

        with torch.amp.autocast(self.device_type, dtype=self.dtype):
            dur_pred = self.dur_model.infer(
                ph_pred.to(self.device), {'tone': tone_pred.to(self.device)},
                None, None, None,
                incremental_state=incremental_state_dur,
                first_decoder_inp=last_dur_token.to(self.device),
                spk_pos_ids_flat=dur_spk_pos_ids_flat.to(self.device),
            )

        dur_pred = dur_pred - 1
        dur_pred = dur_pred.clamp(0, self.dur_model.config['dur_code_size'] - 1)
        if not is_final:
            # add 0.32ms for crossfade
            dur_pred[:, -1] = dur_pred[:, -1] + 32
        else:
            dur_pred[:, -1] = dur_pred[:, -1].clamp(64, 128)

        ''' DiT target speech generation '''
        dur_disturb_choice = (torch.rand_like(dur_pred.float()) > 0.5).float()
        dur_disturb_r = 1 + torch.rand_like(dur_pred.float()) * dur_disturb
        dur_pred = dur_pred * dur_disturb_r * dur_disturb_choice + \
                   dur_pred / dur_disturb_r * (1 - dur_disturb_choice)
        dur_pred = torch.round(dur_pred * dur_alpha).clamp(0, 127)
        # ['。', '！', '？', 'sil']
        for sil_token in [148, 153, 166, 145]:
            dur_pred[ph_pred == sil_token] = dur_pred[ph_pred == sil_token].clamp_min(64)
        # ['，', '；']
        for sil_token in [163, 165]:
            dur_pred[ph_pred == sil_token] = dur_pred[ph_pred == sil_token].clamp_min(32)
        if is_first:
            dur_pred[:, 0] = 8

        dur_sum = dur_pred.sum()
        npad = self.fm - dur_sum % self.fm
        if npad < self.fm:
            dur_pred[:, -1] += npad
        mel2ph_pred = self.length_regulator(dur_pred)
        return mel2ph_pred.detach().cpu()

    def prepare_inputs_for_dit(self, mel2ph_ref, mel2ph_pred, ph_ref, tone_ref, ph_pred, tone_pred, vae_latent):
        # Prepare duration token
        mel2ph_pred = torch.cat((mel2ph_ref, mel2ph_pred + ph_ref.size(1)), dim=1)
        mel2ph_pred = mel2ph_pred[:, :mel2ph_pred.size(1) // self.fm * self.fm].repeat(3, 1)
        # Prepare phone and tone token
        ph_pred = torch.cat((ph_ref, ph_pred), dim=1)
        tone_pred = torch.cat((tone_ref, tone_pred), dim=1)
        # Disable the English tone (set them to 3)"""
        en_tone_idx = ~((tone_pred == 4) | ((11 <= tone_pred) & (tone_pred <= 15)) | (tone_pred == 0))
        tone_pred[en_tone_idx] = 3

        # Prepare cfg inputs
        ph_seq = torch.cat(
            [ph_pred.to(self.device), ph_pred.to(self.device),
             torch.full(ph_pred.size(), self.cfg_mask_token_phone, device=self.device)], 0)
        tone_seq = torch.cat(
            [tone_pred.to(self.device), tone_pred.to(self.device),
             torch.full(tone_pred.size(), self.cfg_mask_token_tone, device=self.device)], 0)
        target_size = mel2ph_pred.size(1) // self.vae_stride
        vae_latent_ = vae_latent.repeat(3, 1, 1)
        ctx_mask = torch.ones_like(vae_latent_[:, :, 0:1])
        vae_latent_ = F.pad(vae_latent_, (0, 0, 0, target_size - vae_latent.size(1)), mode='constant', value=0)
        vae_latent_[1:] = 0.0
        ctx_mask = F.pad(ctx_mask, (0, 0, 0, target_size - vae_latent.size(1)), mode='constant', value=0)

        return {
            'phone': ph_seq,
            'tone': tone_seq,
            "lat_ctx": (vae_latent_ * ctx_mask).to(self.device),
            "ctx_mask": ctx_mask.to(self.device),
            "dur": mel2ph_pred.to(self.device),
        }

    @torch.no_grad()
    def decode(
            self,
            mel2ph_ref,
            mel2ph_pred,
            ph_ref,
            tone_ref,
            ph_pred,
            tone_pred,
            vae_latent,
            loudness_prompt,
            time_step: int = 32,
            p_w: float = 1.6,
            t_w: float = 2.5,
    ) -> np.ndarray:
        dit_inputs = self.prepare_inputs_for_dit(
            mel2ph_ref, mel2ph_pred, ph_ref, tone_ref, ph_pred, tone_pred, vae_latent
        )
        with torch.amp.autocast(self.device_type, dtype=self.dtype):
            x = self.dit.inference(dit_inputs, timesteps=time_step, seq_cfg_w=[p_w, t_w]).float()

        # WavVAE decode
        x[:, :vae_latent.size(1)] = vae_latent
        wav_pred = self.wavvae.decode(x)[0, 0]

        ''' Post-processing '''
        # Trim prompt wav
        wav_pred = wav_pred[vae_latent.size(1) * self.vae_stride * self.hop_size:].detach().cpu()
        if wav_pred.dtype == torch.bfloat16 or wav_pred.dtype == torch.float16:
            wav_pred = wav_pred.to(torch.float32)
        wav_pred = wav_pred.numpy()
        # Norm generated wav to prompt wav's level
        loudness_pred = self.loudness_meter.integrated_loudness(wav_pred.astype(float))
        wav_pred = pyln.normalize.loudness(wav_pred, loudness_pred, loudness_prompt)
        if np.abs(wav_pred).max() >= 1:
            wav_pred = wav_pred / np.abs(wav_pred).max() * 0.95
        return wav_pred
