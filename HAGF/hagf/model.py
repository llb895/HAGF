"""Manuscript-concordant HAGF model used for the APBC 2026 revision.

This implementation follows Eqs. (4)-(11) in the accepted manuscript while
keeping every architectural component independently switchable for ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def entmax_bisect(
    logits: torch.Tensor,
    alpha: float = 1.1,
    dim: int = -1,
    n_iter: int = 20,
) -> torch.Tensor:
    """Compute alpha-entmax probabilities by bisection."""

    if not 1.0 < alpha <= 2.0:
        raise ValueError(f"alpha must be in (1, 2], got {alpha}")

    shifted = logits - logits.amax(dim=dim, keepdim=True)
    scaled = (alpha - 1.0) * shifted
    tau_lo = scaled.amin(dim=dim, keepdim=True) - 1.0
    tau_hi = scaled.amax(dim=dim, keepdim=True)
    power = 1.0 / (alpha - 1.0)

    for _ in range(n_iter):
        tau_mid = (tau_lo + tau_hi) / 2.0
        probs = torch.clamp(scaled - tau_mid, min=0.0).pow(power)
        mass = probs.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(mass > 1.0, tau_mid, tau_lo)
        tau_hi = torch.where(mass > 1.0, tau_hi, tau_mid)

    probs = torch.clamp(scaled - tau_hi, min=0.0).pow(power)
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)


@dataclass(frozen=True)
class AblationConfig:
    use_grouping: bool = True
    use_sparse_masks: bool = True
    use_transformer: bool = True
    use_positional_embedding: bool = True
    use_fidelity_path: bool = True
    use_cross_modal_fusion: bool = True


class DynamicFeatureGroupingLayer(nn.Module):
    """Sparse group-token learning, contextualization, and reconstruction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_masks: int,
        group_ratio: float = 0.2,
        num_heads: int = 4,
        num_transformer_layers: int = 1,
        dropout: float = 0.1,
        entmax_alpha: float = 1.1,
        ablation: AblationConfig | None = None,
    ) -> None:
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be positive")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_masks < 1:
            raise ValueError("num_masks must be positive")
        if not 0.0 < group_ratio <= 1.0:
            raise ValueError("group_ratio must be in (0, 1]")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_masks = num_masks
        self.entmax_alpha = entmax_alpha
        self.ablation = ablation or AblationConfig()

        effective_ratio = group_ratio if self.ablation.use_grouping else 1.0
        self.base_group_size = max(1, int(input_size * effective_ratio))
        self.group_sizes = self._make_group_sizes(input_size, self.base_group_size)
        self.num_groups = len(self.group_sizes)

        self.mask_logits = nn.ParameterList(
            [nn.Parameter(torch.randn(num_masks, width)) for width in self.group_sizes]
        )
        self.gate_transforms = nn.ModuleList(
            [nn.Linear(width, hidden_size, bias=False) for width in self.group_sizes]
        )
        self.value_transforms = nn.ModuleList(
            [nn.Linear(width, hidden_size, bias=False) for width in self.group_sizes]
        )
        self.gate_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in self.group_sizes]
        )
        self.value_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in self.group_sizes]
        )

        self.group_pos_embedding = nn.Parameter(
            torch.randn(1, self.num_groups, hidden_size) * 0.02
        )
        if self.ablation.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.group_transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_transformer_layers,
                enable_nested_tensor=False,
            )
        else:
            self.group_transformer = nn.Identity()
        self.group_projections = nn.ModuleList(
            [nn.Linear(hidden_size, width) for width in self.group_sizes]
        )
        self.last_masks: list[torch.Tensor] = []

    @staticmethod
    def _make_group_sizes(input_size: int, group_size: int) -> list[int]:
        sizes = []
        remaining = input_size
        while remaining > 0:
            width = min(group_size, remaining)
            sizes.append(width)
            remaining -= width
        return sizes

    def normalized_masks(self) -> list[torch.Tensor]:
        if not self.ablation.use_sparse_masks:
            return [
                torch.full_like(logits, 1.0 / logits.shape[-1])
                for logits in self.mask_logits
            ]

        masks: list[torch.Tensor | None] = [None] * self.num_groups
        groups_by_width: dict[int, list[int]] = {}
        for group_index, logits in enumerate(self.mask_logits):
            groups_by_width.setdefault(logits.shape[-1], []).append(group_index)
        for group_indices in groups_by_width.values():
            batched_logits = torch.stack(
                [self.mask_logits[index] for index in group_indices], dim=0
            )
            batched_masks = entmax_bisect(
                batched_logits, alpha=self.entmax_alpha, dim=-1
            )
            for batch_index, group_index in enumerate(group_indices):
                masks[group_index] = batched_masks[batch_index]
        return [mask for mask in masks if mask is not None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(
                f"expected [batch, {self.input_size}], got {tuple(x.shape)}"
            )

        masks_by_group = self.normalized_masks()
        self.last_masks = [mask.detach().cpu() for mask in masks_by_group]
        tokens = []
        start = 0
        for group_idx, width in enumerate(self.group_sizes):
            x_group = x[:, start : start + width]
            start += width
            weighted = x_group.unsqueeze(1) * masks_by_group[group_idx].unsqueeze(0)
            gate = torch.sigmoid(
                self.gate_norms[group_idx](self.gate_transforms[group_idx](weighted))
            )
            value = self.value_norms[group_idx](
                self.value_transforms[group_idx](weighted)
            )
            group_token = F.relu(gate * value).sum(dim=1)
            tokens.append(group_token)

        group_tokens = torch.stack(tokens, dim=1)
        if self.ablation.use_positional_embedding:
            group_tokens = group_tokens + self.group_pos_embedding
        if self.num_groups > 1:
            group_tokens = self.group_transformer(group_tokens)

        reconstructed = [
            projection(group_tokens[:, idx, :])
            for idx, projection in enumerate(self.group_projections)
        ]
        return torch.cat(reconstructed, dim=1)

    def concatenated_masks(self) -> torch.Tensor:
        masks = self.normalized_masks()
        return torch.stack(
            [
                torch.cat([group[mask_idx] for group in masks], dim=0)
                for mask_idx in range(self.num_masks)
            ],
            dim=0,
        )


class HierarchicalFeatureExtractor(nn.Module):
    """Stack grouping layers and retain an original-feature fidelity path."""

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        num_masks: int,
        group_ratio: float,
        num_heads: int = 4,
        entmax_alpha: float = 1.1,
        ablation: AblationConfig | None = None,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.ablation = ablation or AblationConfig()
        self.layers = nn.ModuleList(
            [
                DynamicFeatureGroupingLayer(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_masks=num_masks,
                    group_ratio=group_ratio,
                    num_heads=num_heads,
                    dropout=dropout,
                    entmax_alpha=entmax_alpha,
                    ablation=self.ablation,
                )
                for _ in range(num_layers)
            ]
        )
        self.deep_projection = nn.Linear(input_size, output_size)
        self.fidelity_projection = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)

    @property
    def embedding_size(self) -> int:
        return self.output_size * (2 if self.ablation.use_fidelity_path else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x
        for layer in self.layers:
            x = self.dropout(F.gelu(layer(x)))
        deep = self.deep_projection(x)
        if not self.ablation.use_fidelity_path:
            return deep
        fidelity = F.gelu(self.fidelity_projection(original))
        return torch.cat([deep, fidelity], dim=1)


class CrossModalFusionClassifier(nn.Module):
    """Independent modality branches followed by a fusion classifier."""

    def __init__(
        self,
        input_sizes: Sequence[int],
        num_layers: int,
        hidden_size: int,
        output_size: int,
        num_masks: int,
        group_ratio: float = 0.2,
        dropout: float = 0.1,
        num_heads: int = 4,
        entmax_alpha: float = 1.1,
        ablation: AblationConfig | None = None,
    ) -> None:
        super().__init__()
        if not input_sizes:
            raise ValueError("at least one modality is required")
        self.output_size = output_size
        self.ablation = ablation or AblationConfig()
        self.branches = nn.ModuleList(
            [
                HierarchicalFeatureExtractor(
                    input_size=size,
                    num_layers=num_layers,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    dropout=dropout,
                    num_masks=num_masks,
                    group_ratio=group_ratio,
                    num_heads=num_heads,
                    entmax_alpha=entmax_alpha,
                    ablation=self.ablation,
                )
                for size in input_sizes
            ]
        )
        fusion_size = sum(branch.embedding_size for branch in self.branches)
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )
        self.branch_classifiers = nn.ModuleList(
            [nn.Linear(branch.embedding_size, output_size) for branch in self.branches]
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != len(self.branches):
            raise ValueError(
                f"expected {len(self.branches)} modalities, got {len(inputs)}"
            )
        embeddings = [branch(x) for branch, x in zip(self.branches, inputs)]
        if self.ablation.use_cross_modal_fusion:
            return self.fusion_classifier(torch.cat(embeddings, dim=1))
        logits = [
            head(embedding)
            for head, embedding in zip(self.branch_classifiers, embeddings)
        ]
        return torch.stack(logits, dim=0).mean(dim=0)

    def mask_tensors(self) -> Iterable[tuple[int, int, torch.Tensor]]:
        for branch_idx, branch in enumerate(self.branches):
            for layer_idx, layer in enumerate(branch.layers):
                yield branch_idx, layer_idx, layer.concatenated_masks().detach().cpu()
