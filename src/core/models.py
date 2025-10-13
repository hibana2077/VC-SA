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

from .components import SQuaReFuse


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
    # SQuaRe-Fuse hyperparameters (compat arg names)
    frieren_num_dirs: int = 8,
    frieren_beta: float = 0.5,
    frieren_ortho: bool = True,
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
        # Selection removed; frame_topk/token_topk kept for CLI backward-compat but unused
        # Replace previous fuse with SQuaRe-Fuse
        # Map compat args: frieren_num_dirs -> num_dirs, frieren_beta -> beta_init
        self.fusion = SQuaReFuse(
            d_model=d_model,
            num_dirs=frieren_num_dirs,
            quantiles=(0.1, 0.5, 0.9),
            poly_order=2,
            beta_init=frieren_beta,
            ortho_every_forward=frieren_ortho,
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, 2*d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2*d_model, num_classes),
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
    
    def _validate_hyperparameters(self, *args, **kwargs):
        """Deprecated: selection removed (kept for BC)."""
        return None
    
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
        
        # Temporal fusion directly on full token grid (no selection)
        h = self.fusion(tokens)
        
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
        # Log current beta value of SQuaRe-Fuse
        try:
            beta_val = float(self.fusion.beta.detach().cpu())
            self.log('square/beta', beta_val, prog_bar=False, on_step=True, on_epoch=True)
        except Exception:
            pass
        
        return loss

    def _grad_global_norm(self) -> float:
        total = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total += param_norm.item() ** 2
        return total ** 0.5

    def on_after_backward(self):
        # 记录未裁剪前的梯度范数（如果你还想保留）
        pre = self._grad_global_norm()
        self.log('train/grad_norm_pre_clip', pre, on_step=True, prog_bar=True)

    def on_after_optimizer_step(self, optimizer):
        # 这里已经裁剪完成
        post = self._grad_global_norm()
        self.log('train/grad_norm_post_clip', post, on_step=True, prog_bar=False)
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        clip, label = batch
        logits = self(clip)
        loss = self.criterion(logits, label)
        acc = (logits.argmax(dim=-1) == label).float().mean()

        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        self.log('val/acc', acc, prog_bar=True, on_epoch=True)
        # Log current beta value of SQuaRe-Fuse
        try:
            beta_val = float(self.fusion.beta.detach().cpu())
            self.log('square/beta', beta_val, prog_bar=False, on_epoch=True)
        except Exception:
            pass
        
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
        # Log current beta (epoch-level)
        try:
            beta_val = float(self.fusion.beta.detach().cpu())
            self.log('square/beta', beta_val, on_epoch=True, prog_bar=False)
        except Exception:
            pass

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

    def on_validation_epoch_end(self):
        """Log SQuaRe-Fuse beta at validation epoch end."""
        try:
            beta_val = float(self.fusion.beta.detach().cpu())
            self.log('square/beta', beta_val, on_epoch=True, prog_bar=False)
        except Exception:
            pass


class GraphSamplerActionModelNoSquare(L.LightningModule):
    """
    Ablation variant of the action model without SQuaRe-Fuse.

    It keeps the same backbone and classifier but replaces the fusion block
    with simple mean pooling over time and tokens.

    Constructor signature matches GraphSamplerActionModel for CLI drop-in.
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
        # Compatibility args (ignored here)
        frieren_num_dirs: int = 8,
        frieren_beta: float = 0.5,
        frieren_ortho: bool = True,
    ):
        super().__init__()
        # Save only relevant hparams to avoid exposing unused frieren_* in checkpoints
        self.save_hyperparameters(
            {
                'num_classes': num_classes,
                'frames_per_clip': frames_per_clip,
                'frame_topk': frame_topk,
                'token_topk': token_topk,
                'vit_name': vit_name,
                'vit_pretrained': vit_pretrained,
                'lr': lr,
                'weight_decay': weight_decay,
                'freeze_backbone': freeze_backbone,
                'tau_frame': tau_frame,
                'tau_token': tau_token,
                'graph_knn': graph_knn,
                'graph_tw': graph_tw,
                'graph_layers': graph_layers,
                'use_gat': use_gat,
                'label_smoothing': label_smoothing,
                'test_each_epoch': test_each_epoch,
            }
        )

        # Backbone
        self.backbone = ViTTokenBackbone(
            vit_name,
            vit_pretrained,
            freeze_backbone,
        )

        # Dimensions
        d_model, _ = self._probe_backbone_dims()

        # Classifier identical to main model
        self.cls_head = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d_model, num_classes),
        )

        # Loss
        self.criterion = (
            nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            if label_smoothing > 0
            else nn.CrossEntropyLoss()
        )

        self.test_each_epoch = test_each_epoch

    def _probe_backbone_dims(self) -> Tuple[int, int]:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            tokens = self.backbone(dummy)
            d_model = tokens.shape[-1]
            n_tokens = tokens.shape[1]
        return d_model, n_tokens

    def configure_optimizers(self):
        lr = self.hparams.lr
        wd = self.hparams.weight_decay
        param_groups = [
            {
                'params': [
                    p for n, p in self.named_parameters()
                    if p.requires_grad and 'backbone' in n
                ],
                'lr': lr * 0.5,
                'weight_decay': wd,
            },
            {
                'params': [
                    p for n, p in self.named_parameters()
                    if p.requires_grad and 'backbone' not in n
                ],
                'lr': lr,
                'weight_decay': wd,
            },
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = clip.shape
        x = clip.view(B * T, C, H, W)
        tokens = self.backbone(x)  # [B*T, N, D]
        N = tokens.shape[1]
        D = tokens.shape[2]
        tokens = tokens.view(B, T, N, D)
        # No fusion: simple mean pooling over time and tokens
        feat = tokens.mean(dim=(1, 2))  # [B, D]
        logits = self.cls_head(feat)
        return logits

    def training_step(self, batch, batch_idx):
        clip, label = batch
        logits = self(clip)
        loss = self.criterion(logits, label)
        acc = (logits.argmax(dim=-1) == label).float().mean()
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _grad_global_norm(self) -> float:
        total = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total += param_norm.item() ** 2
        return total ** 0.5

    def on_after_backward(self):
        pre = self._grad_global_norm()
        self.log('train/grad_norm_pre_clip', pre, on_step=True, prog_bar=True)

    def on_after_optimizer_step(self, optimizer):
        post = self._grad_global_norm()
        self.log('train/grad_norm_post_clip', post, on_step=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        clip, label = batch
        logits = self(clip)
        loss = self.criterion(logits, label)
        acc = (logits.argmax(dim=-1) == label).float().mean()
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        self.log('val/acc', acc, prog_bar=True, on_epoch=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        clip, label = batch
        logits = self(clip)
        loss = self.criterion(logits, label)
        acc = (logits.argmax(dim=-1) == label).float().mean()
        self.log('test/loss', loss, prog_bar=False, on_epoch=True)
        self.log('test/acc', acc, prog_bar=False, on_epoch=True)
        return {'test_loss': loss, 'test_acc': acc}

    def on_train_epoch_end(self):
        if not self.test_each_epoch:
            return
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

    def on_validation_epoch_end(self):
        # No square/beta metric for this ablation model
        return None
