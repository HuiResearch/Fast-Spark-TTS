# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/16 14:01
# Author  : Hui Huang
# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear

from .layers import Embedding, LayerNorm, PosEmb
from .transformer import SinusoidalPositionalEmbedding
from .rel_transformer import RelTransformerEncoder

FS_ENCODERS = {
    'rel_fft': lambda hp, dict_size: RelTransformerEncoder(
        dict_size, hp['hidden_size'], hp['hidden_size'],
        hp['ffn_hidden_size'], hp['num_heads'], hp['enc_layers'],
        hp['enc_ffn_kernel_size'], hp['dropout'], prenet=hp['enc_prenet'], pre_ln=hp['enc_pre_ln']),
}


def fill_with_neg_inf2(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(-1e8).type_as(t)


def expand_states(h, mel2token):
    h = F.pad(h, [0, 0, 1, 0])
    mel2token_ = mel2token[..., None].repeat([1, 1, h.shape[-1]])
    h = torch.gather(h, 1, mel2token_)  # [B, T, H]
    return h


class CodePredictor(nn.Module):
    def __init__(self, hparams, hidden_size, dec_hidden_size, dict_size, code_size):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.hparams['hidden_size'] = hidden_size
        self.hidden_size = hidden_size
        char_dict_size = hparams.get('char_dict_size', 4000)
        if not hparams.get('lm_use_enc'):
            self.encoder = nn.Embedding(dict_size, self.hidden_size, padding_idx=0)
            if hparams.get('mega_use_char', True):
                self.char_encoder = nn.Embedding(char_dict_size,
                                                 self.hidden_size, padding_idx=0)
        else:
            self.encoder = FS_ENCODERS[self.hparams['encoder_type']](self.hparams, dict_size)
            if hparams.get('mega_use_char', True):
                self.char_encoder = FS_ENCODERS[self.hparams['encoder_type']](self.hparams, char_dict_size)
            if hparams['use_ph_pos_embed']:
                self.ph_pos_embed = PosEmb(self.hidden_size)

        self.char_empty_embed = nn.Embedding(1, self.hidden_size)
        if hparams.get('use_bert_input'):
            self.bert_input_proj = nn.Linear(768, self.hidden_size)
        self.ling_label_embed_layers = nn.ModuleDict()
        for k, s in zip(hparams['ling_labels'], hparams['ling_label_dict_size']):
            self.ling_label_embed_layers[k] = Embedding(s + 3, self.hidden_size, padding_idx=0)

        self.dec_hidden_size = dec_hidden_size
        self.enc_proj = nn.Linear(self.hidden_size, dec_hidden_size)
        self.code_emb = Embedding(code_size + 2, dec_hidden_size, 0)
        self.use_pos_embed = hparams.get('use_pos_embed', False)
        if self.use_pos_embed:
            self.embed_positions = SinusoidalPositionalEmbedding(dec_hidden_size, 0, init_size=1024)
        self.use_post_ln = hparams.get('use_post_ln', False)
        self.layers = None
        if not self.use_post_ln:
            self.layer_norm = LayerNorm(dec_hidden_size)
        self.code_size = code_size
        self.project_out_dim = Linear(dec_hidden_size, code_size + 1, bias=True)

    def forward_ling_encoder(
            self, txt_tokens, ling_feas, char_tokens, ph2char, bert_embed, spk_id, spk_embed, mels_timbre):
        ph_tokens = txt_tokens
        hparams = self.hparams
        ph_nonpadding = (ph_tokens > 0).float()[:, :, None]  # [B, T_phone, 1]
        x_spk = self.forward_style_embed(spk_embed, spk_id, mels_timbre)

        # enc_ph
        if not hparams.get('lm_use_enc'):
            x_ph = self.encoder(ph_tokens)
            x_ph = x_ph + sum(
                [self.ling_label_embed_layers[k](ling_feas[k]) for k in hparams['ling_labels']]) \
                if len(hparams['ling_labels']) > 0 else 0
            x_ph = x_ph + x_spk
        else:
            # enc_ph
            ph_enc_oembed = sum(
                [self.ling_label_embed_layers[k](ling_feas[k]) for k in hparams['ling_labels']]) \
                if len(hparams['ling_labels']) > 0 else 0
            ph_enc_oembed = ph_enc_oembed + self.ph_pos_embed(
                torch.arange(0, ph_tokens.shape[1])[None,].to(ph_tokens.device))
            ph_enc_oembed = ph_enc_oembed + x_spk
            ph_enc_oembed = ph_enc_oembed * ph_nonpadding
            x_ph = self.encoder(ph_tokens, other_embeds=ph_enc_oembed)

        # enc_char
        if char_tokens is not None and ph2char is not None:
            char_nonpadding = (char_tokens > 0).float()[:, :, None]
            x_char = self.char_encoder(char_tokens)
            empty_char = (ph2char > 100000).long()
            ph2char = ph2char * (1 - empty_char)
            x_char_phlevel = \
                expand_states(x_char * char_nonpadding, ph2char) \
                * (1 - empty_char)[..., None] + \
                self.char_empty_embed(torch.zeros_like(ph_tokens)) * empty_char[..., None]
        else:
            x_char_phlevel = 0
        # x_ling
        x_ling = x_ph + x_char_phlevel
        x_ling = x_ling * ph_nonpadding
        x_ling = self.enc_proj(x_ling)
        return x_ling

    def sample_one_step(self, vq_pred):
        hparams = self.hparams
        if hparams.get('infer_top_k'):
            top_k = hparams.get('infer_top_k')
            temperature = hparams.get('infer_temperature', 1)
            vq_pred = vq_pred[:, -1] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(vq_pred, min(top_k, vq_pred.size(-1)))
                vq_pred[vq_pred < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(vq_pred, dim=-1)
            # sample from the distribution
            vq_pred = torch.multinomial(probs, num_samples=1)
        else:
            vq_pred = torch.argmax(F.softmax(vq_pred[:, -1], dim=-1), 1)
        return vq_pred

    def forward_style_embed(self, spk_embed=None, spk_id=None, mel_ref=None):
        # add spk embed
        style_embed = 0
        if self.hparams['use_spk_embed']:
            style_embed = style_embed + self.spk_embed_proj(spk_embed)[:, None, :]
        if self.hparams['use_spk_id']:
            style_embed = style_embed + self.spk_id_proj(spk_id)[:, None, :]
        if self.hparams['use_spk_enc']:
            style_embed = style_embed + self.spk_enc(mel_ref)[:, None, :]
        return style_embed

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, '_future_mask')
                or self._future_mask is None
                or self._future_mask.device != tensor.device
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(fill_with_neg_inf2(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]
