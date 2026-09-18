"""Binary document classifier and masked focal loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers import AutoModel, PreTrainedModel


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        binary_loss = functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probability = torch.exp(-binary_loss)
        if self.alpha is None:
            alpha_weight: torch.Tensor | float = 0.5
        else:
            alpha = self.alpha.to(logits.device)
            alpha_weight = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_weight * (1 - probability) ** self.gamma * binary_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BinaryDocumentClassifier(PreTrainedModel):
    """One-output classifier with recovered document pooling behavior."""

    supports_gradient_checkpointing = True

    def __init__(
        self,
        config,
        branch: str = "standard",
        positive_alpha: float = 0.5,
    ) -> None:
        super().__init__(config)
        self.branch = branch
        self.encoder = AutoModel.from_config(config)
        dropout = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()
        alpha = torch.tensor([positive_alpha], dtype=torch.float)
        self.register_buffer("positive_alpha", alpha)
        self.loss_fct = FocalLoss(gamma=2.0, alpha=self.positive_alpha)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        global_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        if self.branch == "long_context":
            encoder_args = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            if self.encoder.config.model_type == "longformer":
                encoder_args["global_attention_mask"] = global_attention_mask
            outputs = self.encoder(**encoder_args)
            pooled = (
                outputs.pooler_output
                if getattr(outputs, "pooler_output", None) is not None
                else outputs.last_hidden_state[:, 0, :]
            )
            logits = self.classifier(self.dropout(pooled))
        else:
            batch_size, chunk_count, sequence_length = input_ids.size()
            outputs = self.encoder(
                input_ids=input_ids.view(-1, sequence_length),
                attention_mask=attention_mask.view(-1, sequence_length),
                token_type_ids=(
                    token_type_ids.view(-1, sequence_length)
                    if token_type_ids is not None
                    else None
                ),
            )
            pooled = outputs[1]
            chunk_logits = self.classifier(self.dropout(pooled)).view(
                batch_size, chunk_count, 1
            )
            logits = torch.mean(chunk_logits, dim=1)

        if labels is None:
            return logits
        mask = (labels != -100).float()
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0
        self.loss_fct.reduction = "none"
        sample_loss = self.loss_fct(logits, safe_labels)
        loss = (sample_loss * mask).sum() / (mask.sum() + 1e-9)
        return loss, logits

