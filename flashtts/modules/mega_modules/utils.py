# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/16 15:17
# Author  : Hui Huang
import torch


def mel2token_to_dur(mel2token, T_txt=None, max_dur=None):
    is_torch = isinstance(mel2token, torch.Tensor)
    has_batch_dim = True
    if not is_torch:
        mel2token = torch.LongTensor(mel2token)
    if T_txt is None:
        T_txt = mel2token.max()
    if len(mel2token.shape) == 1:
        mel2token = mel2token[None, ...]
        has_batch_dim = False
    B, _ = mel2token.shape
    dur = mel2token.new_zeros(B, T_txt + 1).scatter_add(1, mel2token, torch.ones_like(mel2token))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    if not is_torch:
        dur = dur.numpy()
    if not has_batch_dim:
        dur = dur[0]
    return dur
