#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal Video Action Recognition training pipeline using:
  - PyTorch Lightning (for training loop / logging)
  - torchvision / timm (ViT image backbone pretrained on images)
  - example.core (FrameTokenCoSelector & GraphBasedMemBank) for efficient frame+token selection & temporal graph memory

Features:
  * CLI control (see --help)
  * On-the-fly decoding of .mp4 / .avi -> (optional) frame cache -> uniform temporal sampling
  * Frame+token co-selection before temporal modeling (GraphBasedMemBank)
  * Test split evaluated automatically at the end of every training epoch (without invoking Trainer.test to avoid nested loops)
  * Mixed precision, gradient accumulation, checkpointing, and optional backbone freezing

Expected annotation CSV format (train / val / test):
	video_path,label
Where video_path is absolute OR relative to --data-root. Labels should be integer class ids [0, num_classes-1].

Example usage:
  python -m src.run \
	  --data-root data/videos \
	  --train-anno train.csv --val-anno val.csv --test-anno test.csv \
	  --num-classes 400 --frames-per-clip 16 --frame-topk 8 --token-topk 32 \
	  --batch-size 2 --max-epochs 50 --lr 5e-4 --accumulate 2

Notes:
  * This is an MVP; for large-scale training consider replacing naive per-sample decoding with batched decoders (PyAV, decord) and multi-epoch frame caching.
  * GraphBasedMemBank is currently O(B * T * k * D) with Python-side loops; further vectorization can improve throughput.
  * Data augmentations here are minimal (resize + center/random crop placeholder). Extend as needed.
"""

from __future__ import annotations

import os
import csv
import math
import time
import argparse
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.io import read_video

import timm

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

# Allow importing example.core while keeping current repo layout
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from example.core import FrameTokenCoSelector, GraphBasedMemBank  # noqa: E402


# -----------------------------
# Data Utilities
# -----------------------------

def _hash_path(p: str) -> str:
	return hashlib.sha1(p.encode('utf-8')).hexdigest()[:16]


class VideoRecord:
	def __init__(self, path: str, label: int):
		self.path = path
		self.label = label


class VideoCSVAnnotation:
	def __init__(self, csv_path: str, data_root: Optional[str] = None):
		self.records: List[VideoRecord] = []
		csv_path = Path(csv_path)
		if not csv_path.is_file():
			raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")
		with csv_path.open('r', newline='', encoding='utf-8') as f:
			reader = csv.reader(f)
			for row in reader:
				if not row or row[0].startswith('#'):
					continue
				if len(row) < 2:
					raise ValueError(f"CSV row must have at least 2 columns (path,label): {row}")
				vp = row[0]
				if data_root and not os.path.isabs(vp):
					vp = os.path.join(data_root, vp)
				label = int(row[1])
				self.records.append(VideoRecord(vp, label))
		if len(self.records) == 0:
			raise ValueError(f"No valid records in {csv_path}")

	def __len__(self):
		return len(self.records)

	def __getitem__(self, idx: int) -> VideoRecord:
		return self.records[idx]


class FrameCache:
	"""Simple disk frame cache. Given a video path, stores frames as JPEGs inside cache_root/<hash>/frame_%05d.jpg.
	Caches only if not already cached. Returns list[Path] of frame files.
	"""
	def __init__(self, cache_root: Optional[str]):
		self.cache_root = Path(cache_root) if cache_root else None
		if self.cache_root:
			self.cache_root.mkdir(parents=True, exist_ok=True)

	def get_or_extract(self, video_path: str) -> List[Path]:
		if self.cache_root is None:
			return []  # Signal to decode on-the-fly
		vid_hash = _hash_path(video_path)
		out_dir = self.cache_root / vid_hash
		frame_files: List[Path] = []
		if out_dir.is_dir():
			frame_files = sorted(out_dir.glob('frame_*.jpg'))
			if frame_files:
				return frame_files
		# Extract
		out_dir.mkdir(parents=True, exist_ok=True)
		try:
			video, _, _ = read_video(video_path, pts_unit='sec')  # [T, H, W, C] uint8
		except Exception as e:
			raise RuntimeError(f"Failed to read video {video_path}: {e}")
		for i in range(video.shape[0]):
			frame = video[i].numpy()
			img = torchvision.transforms.functional.to_pil_image(frame)
			img.save(out_dir / f"frame_{i:05d}.jpg", quality=90)
		frame_files = sorted(out_dir.glob('frame_*.jpg'))
		return frame_files


class SimpleVideoDataset(Dataset):
	def __init__(
		self,
		anno: VideoCSVAnnotation,
		num_frames: int,
		frame_cache: FrameCache,
		is_train: bool = True,
		resize: int = 224,
	):
		self.anno = anno
		self.num_frames = num_frames
		self.frame_cache = frame_cache
		self.is_train = is_train
		self.resize = resize
		self.tx = self._build_transform()

	def _build_transform(self):
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])
		aug: List[transforms.Module] = [transforms.Resize((self.resize, self.resize))]
		if self.is_train:
			# Placeholder for more advanced augmentation
			pass
		aug.extend([
			transforms.ToTensor(),
			normalize,
		])
		return transforms.Compose(aug)

	def _sample_indices(self, total: int) -> List[int]:
		if total <= self.num_frames:
			return list(range(total))
		# Uniform sampling
		stride = total / self.num_frames
		return [int(stride * i + stride * 0.5) for i in range(self.num_frames)]

	def _load_frames(self, video_path: str) -> torch.Tensor:
		frame_files = self.frame_cache.get_or_extract(video_path)
		frames: List[torch.Tensor] = []
		if frame_files:  # Use cached frames
			total = len(frame_files)
			idxs = self._sample_indices(total)
			for i in idxs:
				img = torchvision.io.read_image(str(frame_files[i]))  # [C,H,W] uint8
				img = img.float() / 255.0
				# Apply normalization manually because transform pipeline expects PIL
				# We'll convert to PIL earlier instead for consistency
				pil = transforms.functional.to_pil_image(img)
				frames.append(self.tx(pil))
		else:
			# Decode directly and sample
			video, _, _ = read_video(video_path, pts_unit='sec')  # [T,H,W,C]
			total = video.shape[0]
			idxs = self._sample_indices(total)
			for i in idxs:
				frame = video[i]  # [H,W,C]
				pil = transforms.functional.to_pil_image(frame)
				frames.append(self.tx(pil))
		if len(frames) == 0:
			raise RuntimeError(f"No frames extracted for {video_path}")
		clip = torch.stack(frames, dim=0)  # [T, C, H, W]
		return clip

	def __len__(self):
		return len(self.anno)

	def __getitem__(self, idx: int):
		record = self.anno[idx]
		clip = self._load_frames(record.path)  # [T,C,H,W]
		label = record.label
		return clip, label


class VideoDataModule(L.LightningDataModule):
	def __init__(
		self,
		data_root: str,
		train_csv: str,
		val_csv: str,
		test_csv: str,
		frames_per_clip: int,
		batch_size: int,
		num_workers: int = 4,
		frame_cache_dir: Optional[str] = None,
		resize: int = 224,
	):
		super().__init__()
		self.save_hyperparameters()
		self.frame_cache = FrameCache(frame_cache_dir)

	def setup(self, stage: Optional[str] = None):
		h = self.hparams
		if stage in (None, 'fit'):
			self.train_set = SimpleVideoDataset(
				VideoCSVAnnotation(h.train_csv, h.data_root),
				h.frames_per_clip, self.frame_cache, is_train=True, resize=h.resize,
			)
			self.val_set = SimpleVideoDataset(
				VideoCSVAnnotation(h.val_csv, h.data_root),
				h.frames_per_clip, self.frame_cache, is_train=False, resize=h.resize,
			)
		if stage in (None, 'test', 'predict'):
			self.test_set = SimpleVideoDataset(
				VideoCSVAnnotation(h.test_csv, h.data_root),
				h.frames_per_clip, self.frame_cache, is_train=False, resize=h.resize,
			)

	def train_dataloader(self):
		return DataLoader(
			self.train_set,
			batch_size=self.hparams.batch_size,
			shuffle=True,
			num_workers=self.hparams.num_workers,
			pin_memory=True,
			drop_last=True,
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_set,
			batch_size=self.hparams.batch_size,
			shuffle=False,
			num_workers=self.hparams.num_workers,
			pin_memory=True,
		)

	def test_dataloader(self):
		return DataLoader(
			self.test_set,
			batch_size=self.hparams.batch_size,
			shuffle=False,
			num_workers=self.hparams.num_workers,
			pin_memory=True,
		)


# -----------------------------
# Model
# -----------------------------

class ViTTokenBackbone(nn.Module):
	"""Wrap a timm ViT to output patch tokens (excluding cls) with shape [B, N, D]."""
	def __init__(self, model_name: str = 'vit_base_patch16_224', pretrained: bool = True, freeze: bool = False):
		super().__init__()
		self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
		# timm ViT forward_features returns [B, N+1, D] (cls + patches)
		if freeze:
			for p in self.vit.parameters():
				p.requires_grad = False

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, C, H, W]
		feats = self.vit.forward_features(x)  # [B, N+1, D] or [B, N, D] depending on config
		if feats.shape[1] > 0 and hasattr(self.vit, 'cls_token'):
			# Remove CLS token if present
			feats = feats[:, 1:, :]
		return feats  # [B, N, D]


class GraphSamplerActionModel(L.LightningModule):
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
	):
		super().__init__()
		self.save_hyperparameters()
		self.backbone = ViTTokenBackbone(vit_name, vit_pretrained, freeze_backbone)
		# Probe one dummy tensor to infer dims (lazy alternative: specify manually)
		with torch.no_grad():
			dummy = torch.zeros(1, 3, 224, 224)
			tokens = self.backbone(dummy)
			d_model = tokens.shape[-1]
			n_tokens = tokens.shape[1]
		if token_topk > n_tokens:
			raise ValueError(f"token_topk ({token_topk}) > number of ViT patch tokens ({n_tokens})")
		if frame_topk > frames_per_clip:
			raise ValueError(f"frame_topk ({frame_topk}) > frames_per_clip ({frames_per_clip})")

		self.co_selector = FrameTokenCoSelector(
			d_model=d_model,
			frame_topk=frame_topk,
			token_topk=token_topk,
			use_cls=False,
			tau_frame=tau_frame,
			tau_token=tau_token,
		)
		self.graph_mem = GraphBasedMemBank(
			d_model=d_model,
			knn_k=graph_knn,
			temporal_window=graph_tw,
			num_layers=graph_layers,
			use_gat=use_gat,
		)
		self.cls_head = nn.Sequential(
			nn.LayerNorm(d_model),
			nn.Linear(d_model, num_classes)
		)
		self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing) if label_smoothing > 0 else nn.CrossEntropyLoss()
		self.test_each_epoch = test_each_epoch

	def configure_optimizers(self):
		lr = self.hparams.lr
		wd = self.hparams.weight_decay
		param_groups = [
			{'params': [p for n, p in self.named_parameters() if p.requires_grad and 'backbone' in n], 'lr': lr * 0.5, 'weight_decay': wd},
			{'params': [p for n, p in self.named_parameters() if p.requires_grad and 'backbone' not in n], 'lr': lr, 'weight_decay': wd},
		]
		optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
		return {"optimizer": optimizer, "lr_scheduler": scheduler}

	def forward(self, clip: torch.Tensor) -> torch.Tensor:
		# clip: [B, T, C, H, W]
		B, T, C, H, W = clip.shape
		x = clip.view(B * T, C, H, W)
		tokens = self.backbone(x)  # [B*T, N, D]
		N = tokens.shape[1]
		D = tokens.shape[2]
		tokens = tokens.view(B, T, N, D)
		z, frame_idx, token_idx, frame_mask, token_mask = self.co_selector(tokens)
		# Graph memory expects [B, T', M, D] where T' = frame_topk, M = token_topk
		h, _ = self.graph_mem(z, reset_memory=True)
		# Global average over time & tokens
		feat = h.mean(dim=(1, 2))  # [B, D]
		logits = self.cls_head(feat)  # [B, C]
		return logits

	def training_step(self, batch, batch_idx):
		clip, label = batch
		logits = self(clip)
		loss = self.criterion(logits, label)
		acc = (logits.argmax(dim=-1) == label).float().mean()
		self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
		self.log('train/acc', acc, prog_bar=True, on_step=True, on_epoch=True)
		return loss

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
		# Manual test evaluation (no gradient). Avoid calling trainer.test to keep training state clean.
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


# -----------------------------
# CLI / Main
# -----------------------------

def parse_args():
	p = argparse.ArgumentParser(description='Video Action Recognition (GraphSampler)')
	# Data
	p.add_argument('--data-root', type=str, default='.', help='Root to prepend to relative video paths in CSV')
	p.add_argument('--train-anno', type=str, required=True)
	p.add_argument('--val-anno', type=str, required=True)
	p.add_argument('--test-anno', type=str, required=True)
	p.add_argument('--frame-cache', type=str, default=None, help='Directory to cache decoded frames (optional)')
	# Model hyperparameters
	p.add_argument('--num-classes', type=int, required=True)
	p.add_argument('--frames-per-clip', type=int, default=16)
	p.add_argument('--frame-topk', type=int, default=8)
	p.add_argument('--token-topk', type=int, default=32)
	p.add_argument('--vit-name', type=str, default='vit_base_patch16_224')
	p.add_argument('--no-pretrained', action='store_true')
	p.add_argument('--freeze-backbone', action='store_true')
	p.add_argument('--tau-frame', type=float, default=1.0)
	p.add_argument('--tau-token', type=float, default=0.7)
	p.add_argument('--graph-knn', type=int, default=8)
	p.add_argument('--graph-tw', type=int, default=2, help='Temporal window (previous frames) for graph edges')
	p.add_argument('--graph-layers', type=int, default=1)
	p.add_argument('--no-gat', action='store_true')
	# Optimization
	p.add_argument('--batch-size', type=int, default=2)
	p.add_argument('--num-workers', type=int, default=4)
	p.add_argument('--max-epochs', type=int, default=30)
	p.add_argument('--lr', type=float, default=5e-4)
	p.add_argument('--weight-decay', type=float, default=0.05)
	p.add_argument('--label-smoothing', type=float, default=0.0)
	p.add_argument('--accumulate', type=int, default=1, help='Gradient accumulation steps')
	p.add_argument('--precision', type=str, default='16-mixed', choices=['32', '16-mixed', 'bf16-mixed'])
	p.add_argument('--seed', type=int, default=42)
	# Logging / Checkpoint
	p.add_argument('--output', type=str, default='outputs')
	p.add_argument('--project', type=str, default='graphsampler')
	p.add_argument('--test-each-epoch', action='store_true', help='Run test split evaluation each epoch (manual loop)')
	# Hardware
	p.add_argument('--devices', type=int, default=1, help='Number of GPUs (if 0, use CPU)')
	p.add_argument('--strategy', type=str, default='auto')
	return p.parse_args()


def main():
	args = parse_args()
	L.seed_everything(args.seed, workers=True)

	os.makedirs(args.output, exist_ok=True)
	logger = CSVLogger(save_dir=args.output, name=args.project)

	dm = VideoDataModule(
		data_root=args.data_root,
		train_csv=args.train_anno,
		val_csv=args.val_anno,
		test_csv=args.test_anno,
		frames_per_clip=args.frames_per_clip,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		frame_cache_dir=args.frame_cache,
		resize=224,
	)

	model = GraphSamplerActionModel(
		num_classes=args.num_classes,
		frames_per_clip=args.frames_per_clip,
		frame_topk=args.frame_topk,
		token_topk=args.token_topk,
		vit_name=args.vit_name,
		vit_pretrained=not args.no_pretrained,
		lr=args.lr,
		weight_decay=args.weight_decay,
		freeze_backbone=args.freeze_backbone,
		tau_frame=args.tau_frame,
		tau_token=args.tau_token,
		graph_knn=args.graph_knn,
		graph_tw=args.graph_tw,
		graph_layers=args.graph_layers,
		use_gat=not args.no_gat,
		label_smoothing=args.label_smoothing,
		test_each_epoch=args.test_each_epoch,
	)

	ckpt_cb = ModelCheckpoint(
		dirpath=os.path.join(args.output, args.project, 'checkpoints'),
		filename='epoch{epoch:02d}-val_acc{val/acc:.3f}',
		monitor='val/acc',
		mode='max',
		save_top_k=3,
		save_last=True,
		auto_insert_metric_name=False,
	)
	lr_cb = LearningRateMonitor(logging_interval='epoch')

	trainer = L.Trainer(
		max_epochs=args.max_epochs,
		devices=args.devices if args.devices > 0 else None,
		accelerator='gpu' if args.devices > 0 and torch.cuda.is_available() else 'cpu',
		precision=args.precision,
		accumulate_grad_batches=args.accumulate,
		strategy=args.strategy,
		logger=logger,
		callbacks=[ckpt_cb, lr_cb],
		log_every_n_steps=10,
		deterministic=False,
	)

	trainer.fit(model, datamodule=dm)

	# Final test (best or last checkpoint automatically handled if needed)
	trainer.test(model=model, datamodule=dm)


if __name__ == '__main__':
	main()

