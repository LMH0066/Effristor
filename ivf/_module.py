import copy
import math
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi import settings
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch import Tensor
from torch.nn.modules import Dropout, LayerNorm, Linear

from ivf._utils import LOSS_KEYS


class PositionEncoding(nn.Module):
    def __init__(self, max_len, d_model, device):
        super(PositionEncoding, self).__init__()

        self.PE = torch.zeros(max_len, d_model, device=device)
        self.PE.requires_grad = False

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2, device=device).float()

        self.PE[:, 0::2] = torch.sin(pos / (10000**(_2i / d_model)))
        self.PE[:, 1::2] = torch.cos(pos / (10000**(_2i / d_model)))

    def forward(self, x):
        # [batch_size, seq_len]
        seq_len = x.size(1)
        return self.PE[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, dropout, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionEncoding(max_len, d_model, device)
        self.drop_out = nn.Dropout(p=dropout)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, outputs, targets):
        pt = torch.sigmoid(outputs)
        loss = - self.alpha * (1 - pt) ** self.gamma * targets * torch.log(pt) - (
            1 - self.alpha) * pt ** self.gamma * (1 - outputs) * torch.log(1 - pt)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "max":
            loss = loss.max()

        return loss


class NET(BaseModuleClass):
    def __init__(
        self,
        main_loc: str,
        target_loc: str,
        output_dim: int = 1,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        seed: int = 42,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        settings.seed = seed

        self.main_loc, self.target_loc = main_loc, target_loc
        self.num_labels = output_dim

        self.embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            **factory_kwargs,
        )
        encoder_norm = LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=bias),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4, bias=bias),
            nn.ReLU(),
            nn.Linear(d_model // 4, output_dim, bias=bias),
        )

        self._init_weights()

    def _get_inference_input(self, tensors: Dict[Any, Any], **kwargs):
        main = tensors[self.main_loc]  # [batch_size, tokens]

        input_dict = {"main": main}
        return input_dict

    @auto_move_data
    def inference(self, main: torch.Tensor):
        _main = main.squeeze()
        if len(_main.shape) == 1:
            _main = _main.unsqueeze(0)
        attn_logits = self.encoder(
            self.embedding(torch.nan_to_num(
                _main, nan=-255.0).unsqueeze(-1)).permute(1, 0, 2),
            src_key_padding_mask=torch.isnan(_main)
        )
        attn_logits = attn_logits.mean(dim=0)
        out = self.decoder(attn_logits)
        return out

    @auto_move_data
    def loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_output: torch.Tensor,
    ) -> Dict[str, float]:
        losses = dict()
        target = tensors[self.target_loc].squeeze()

        losses[LOSS_KEYS.FocalLoss] = FocalLoss()(
            inference_output.squeeze(), target)

        return losses

    @auto_move_data
    def forward(
        self,
        tensors,
        compute_loss=True,
        get_inference_input_kwargs: Optional[dict] = None,
    ) -> Union[tuple[torch.Tensor], tuple[torch.Tensor, LossOutput]]:
        inference_inputs = self._get_inference_input(tensors)
        inference_output = self.inference(**inference_inputs)

        if compute_loss:
            losses = self.loss(tensors, inference_output)
            return inference_output, losses
        else:
            return inference_output

    # TODO
    def _init_weights(self, initializer_range=0.02):
        """Initialize the weights"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def load_state_dict(self, state_dict):
        def _remove_prefix(text, prefix):
            if text.startswith(prefix):
                return text[len(prefix):]
            return text

        pairings = [
            (src_key, _remove_prefix(src_key, "_orig_mod."))
            for src_key in state_dict.keys()
        ]
        if all(src_key == dest_key for src_key, dest_key in pairings):
            super(NET, self).load_state_dict(state_dict)
            return  # Do not write checkpoint if no need to repair!

        _state_dict = {}
        for src_key, dest_key in pairings:
            _state_dict[dest_key] = state_dict[src_key]
        super(NET, self).load_state_dict(_state_dict)
