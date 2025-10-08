#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model architectures for Video Action Recognition.

This module contains:
  - ViTTokenBackbone: Vision Transformer backbone that outputs patch tokens
  - GraphSamplerActionModel: Main action recognition model with frame/token selection and graph-based temporal modeling
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import lightning as L

from pathlib import Path
import sys

from .components import (
    FrameTokenCoSelector,
    GraphBasedMemBank,
    FPSChangePointSelector,
    LatentCrossAttnMemBank,
    TemporalConvMemBank,
)


class ViTTokenBackbone(nn.Module):
    """
    Vision Transformer backbone wrapper.
    
    Wraps a timm ViT model to output patch tokens (excluding CLS token) 
    with shape [B, N, D] where:
      - B: batch size
      - N: number of patch tokens
      - D: embedding dimension
    
    Args:
        model_name: Name of the timm ViT model to use
        pretrained: Whether to load pretrained weights
        freeze: Whether to freeze backbone parameters
    """
    
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        freeze: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze = freeze
        
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT backbone.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Patch token features of shape [B, N, D]
        """
        # x: [B, C, H, W]
        feats = self.vit.forward_features(x)  # [B, N+1, D] or [B, N, D]
        
        # Remove CLS token if present
        if feats.shape[1] > 0 and hasattr(self.vit, 'cls_token'):
            feats = feats[:, 1:, :]
            
        return feats  # [B, N, D]
    
    def get_feature_dim(self) -> int:
        """Get the output feature dimension."""
        return self.vit.embed_dim if hasattr(self.vit, 'embed_dim') else self.vit.num_features


class GraphSamplerActionModel(L.LightningModule):
    """
    Main video action recognition model.
    
    This model combines:
      1. ViT backbone for frame-level feature extraction
      2. FrameTokenCoSelector for efficient frame and token selection
      3. GraphBasedMemBank for temporal modeling with graph structure
      4. Classification head for action prediction
    
    The model processes videos by:
      - Extracting patch tokens from sampled frames
      - Selecting most informative frames and tokens
      - Building temporal graphs for efficient modeling
      - Aggregating features for final classification
    
    Args:
        num_classes: Number of action classes
        frames_per_clip: Number of frames sampled from each video
        frame_topk: Number of top frames to select
        token_topk: Number of top tokens to select per frame
        vit_name: Name of ViT backbone model
        vit_pretrained: Whether to use pretrained ViT weights
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        freeze_backbone: Whether to freeze backbone during training
        tau_frame: Temperature for frame selection
        tau_token: Temperature for token selection
        graph_knn: Number of nearest neighbors in graph
        graph_tw: Temporal window size for graph edges
        graph_layers: Number of graph convolution layers
        use_gat: Whether to use Graph Attention Network
        label_smoothing: Label smoothing factor
        test_each_epoch: Whether to evaluate test set after each epoch
    """
    
    def __init__(
        self,
        num_classes: int,
        frames_per_clip: int,
        frame_topk: int,
        token_topk: int,
        vit_name: str = 'vit_base_patch16_224',
        vit_pretrained: bool = True,
        lr: float = 5e-4,
        weight_decay: float = 0.05,
        freeze_backbone: bool = False,
        tau_frame: float = 1.0,
        tau_token: float = 0.7,
        graph_knn: int = 8,
        graph_tw: int = 2,
        graph_layers: int = 1,
        use_gat: bool = True,
        label_smoothing: float = 0.0,
        test_each_epoch: bool = True,
        # New options for alternative modules
        selector_type: str = 'learnable',  # 'learnable' | 'fps'
        membank_type: str = 'graph',       # 'graph' | 'latent' | 'tcn'
        latent_slots: int = 64,
        latent_heads: int = 8,
        tcn_kernel: int = 5,
        tcn_dilations: tuple = (1, 2, 4),
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize backbone
        self.backbone = ViTTokenBackbone(
            vit_name,
            vit_pretrained,
            freeze_backbone
        )
        
        # Probe backbone to get dimensions
        d_model, n_tokens = self._probe_backbone_dims()
        
        # Validate hyperparameters
        self._validate_hyperparameters(
            token_topk, n_tokens,
            frame_topk, frames_per_clip
        )
        
        # Initialize frame/token selector (choose type)
        if selector_type.lower() == 'learnable':
            self.co_selector = FrameTokenCoSelector(
                d_model=d_model,
                frame_topk=frame_topk,
                token_topk=token_topk,
                use_cls=False,
                tau_frame=tau_frame,
                tau_token=tau_token,
            )
        elif selector_type.lower() == 'fps':
            self.co_selector = FPSChangePointSelector(
                d_model=d_model,
                frame_topk=frame_topk,
                token_topk=token_topk,
                use_cls=False,
                ema_alpha=0.9,
            )
        else:
            raise ValueError(f"Unsupported selector_type: {selector_type}")

        # Initialize temporal memory bank (choose type)
        if membank_type.lower() == 'graph':
            self.mem_bank = GraphBasedMemBank(
                d_model=d_model,
                knn_k=graph_knn,
                temporal_window=graph_tw,
                num_layers=graph_layers,
                use_gat=use_gat,
            )
        elif membank_type.lower() == 'latent':
            self.mem_bank = LatentCrossAttnMemBank(
                d_model=d_model,
                num_latents=latent_slots,
                num_heads=latent_heads,
            )
        elif membank_type.lower() == 'tcn':
            self.mem_bank = TemporalConvMemBank(
                d_model=d_model,
                kernel_size=tcn_kernel,
                dilations=tcn_dilations,
            )
        else:
            raise ValueError(f"Unsupported membank_type: {membank_type}")
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
        
        # Loss function
        self.criterion = (
            nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            if label_smoothing > 0
            else nn.CrossEntropyLoss()
        )
        
        self.test_each_epoch = test_each_epoch
    
    def _probe_backbone_dims(self) -> Tuple[int, int]:
        """Probe backbone to determine feature dimensions."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            tokens = self.backbone(dummy)
            d_model = tokens.shape[-1]
            n_tokens = tokens.shape[1]
        return d_model, n_tokens
    
    def _validate_hyperparameters(
        self,
        token_topk: int,
        n_tokens: int,
        frame_topk: int,
        frames_per_clip: int
    ):
        """Validate model hyperparameters."""
        if token_topk > n_tokens:
            raise ValueError(
                f"token_topk ({token_topk}) > number of ViT patch tokens ({n_tokens})"
            )
        if frame_topk > frames_per_clip:
            raise ValueError(
                f"frame_topk ({frame_topk}) > frames_per_clip ({frames_per_clip})"
            )
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        lr = self.hparams.lr
        wd = self.hparams.weight_decay
        
        # Separate parameter groups for backbone and other components
        param_groups = [
            {
                'params': [
                    p for n, p in self.named_parameters()
                    if p.requires_grad and 'backbone' in n
                ],
                'lr': lr * 0.5,
                'weight_decay': wd
            },
            {
                'params': [
                    p for n, p in self.named_parameters()
                    if p.requires_grad and 'backbone' not in n
                ],
                'lr': lr,
                'weight_decay': wd
            },
        ]
        
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            clip: Input video clip of shape [B, T, C, H, W]
            
        Returns:
            Logits of shape [B, num_classes]
        """
        B, T, C, H, W = clip.shape
        
        # Extract features from all frames
        x = clip.view(B * T, C, H, W)
        tokens = self.backbone(x)  # [B*T, N, D]
        
        N = tokens.shape[1]
        D = tokens.shape[2]
        tokens = tokens.view(B, T, N, D)
        
        # Select important frames and tokens
        z, frame_idx, token_idx, frame_mask, token_mask = self.co_selector(tokens)
        
        # Apply temporal modeling via selected memory bank
        # z has shape [B, T', M, D] where T' = frame_topk, M = token_topk
        if isinstance(self.mem_bank, GraphBasedMemBank):
            h, _ = self.mem_bank(z, reset_memory=True)
        else:
            h, _ = self.mem_bank(z)
        
        # Global average pooling over time and tokens
        feat = h.mean(dim=(1, 2))  # [B, D]
        
        # Classification
        logits = self.cls_head(feat)  # [B, num_classes]
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        clip, label = batch
        logits = self(clip)
        loss = self.criterion(logits, label)
        acc = (logits.argmax(dim=-1) == label).float().mean()
        
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def on_after_backward(self):
        """Log gradient norms after backward pass."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log('train/grad_norm', total_norm, on_step=True, on_epoch=True, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        clip, label = batch
        logits = self(clip)
        loss = self.criterion(logits, label)
        acc = (logits.argmax(dim=-1) == label).float().mean()
        
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        self.log('val/acc', acc, prog_bar=True, on_epoch=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        clip, label = batch
        logits = self(clip)
        loss = self.criterion(logits, label)
        acc = (logits.argmax(dim=-1) == label).float().mean()
        
        self.log('test/loss', loss, prog_bar=False, on_epoch=True)
        self.log('test/acc', acc, prog_bar=False, on_epoch=True)
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def on_train_epoch_end(self):
        """
        Hook called at the end of training epoch.
        
        Optionally runs test evaluation without invoking Trainer.test
        to avoid nested training loops.
        """
        if not self.test_each_epoch:
            return
        
        # Manual test evaluation (no gradient)
        datamodule = self.trainer.datamodule
        test_loader = datamodule.test_dataloader()
        
        self.eval()
        total, correct, losses = 0, 0, []
        criterion = self.criterion
        
        with torch.no_grad():
            for clip, label in test_loader:
                clip = clip.to(self.device)
                label = label.to(self.device)
                
                logits = self(clip)
                loss = criterion(logits, label)
                losses.append(loss.detach())
                
                pred = logits.argmax(dim=-1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        
        if total > 0:
            test_acc = correct / total
            test_loss = torch.stack(losses).mean().item() if losses else 0.0
            self.log('epoch_test/acc', test_acc, prog_bar=True)
            self.log('epoch_test/loss', test_loss, prog_bar=False)
        
        self.train()
