"""Archived Legacy Components (Read-Only)
========================================

This file preserves the original learnable frame/token co-selector and the
graph-based memory bank plus experimental alternative modules (FPS selector,
latent cross-attention memory, temporal conv memory) prior to migration to the
StatLite versions and SQuaRe-Fuse.

They are no longer used in the main pipeline but retained for experimentation.
Import manually if needed, e.g.:

	from core.components_legacy import FrameTokenCoSelector, GraphBasedMemBank

"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def _topk_straight_through(logits: torch.Tensor, k: int, dim: int = -1, tau: float = 1.0):
	soft = F.softmax(logits / max(tau, 1e-6), dim=dim)
	topk = torch.topk(soft, k=min(k, soft.size(dim)), dim=dim)
	hard = torch.zeros_like(soft)
	hard.scatter_(dim, topk.indices, 1.0)
	mask = hard + (soft - hard).detach()
	return mask, topk.indices


class FrameTokenCoSelector(nn.Module):
	def __init__(
		self,
		d_model: int,
		frame_topk: int,
		token_topk: int,
		use_cls: bool = False,
		tau_frame: float = 1.0,
		tau_token: float = 1.0,
		hidden: int = 4,
	):
		super().__init__()
		self.frame_topk = frame_topk
		self.token_topk = token_topk
		self.tau_frame = tau_frame
		self.tau_token = tau_token
		self.use_cls = use_cls
		self.frame_mlp = nn.Sequential(
			nn.LayerNorm(d_model),
			nn.Linear(d_model, hidden * d_model),
			nn.GELU(),
			nn.Linear(hidden * d_model, 1),
		)
		self.token_mlp = nn.Sequential(
			nn.LayerNorm(d_model),
			nn.Linear(d_model, d_model // 2),
			nn.GELU(),
			nn.Linear(d_model // 2, 1),
		)

	def forward(
		self,
		x: torch.Tensor,
		mask: Optional[torch.Tensor] = None,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		B, T, N, D = x.shape
		if mask is None:
			mask = torch.ones(B, T, N, device=x.device, dtype=x.dtype)
		if self.use_cls and N >= 1:
			frame_repr = x[:, :, 0, :]
		else:
			denom = mask.sum(dim=2, keepdim=False).clamp_min(1e-6)[..., None]
			frame_repr = (x * mask[..., None]).sum(dim=2) / denom
		frame_logit = self.frame_mlp(frame_repr).squeeze(-1)
		frame_mask, frame_idx = _topk_straight_through(frame_logit, self.frame_topk, dim=1, tau=self.tau_frame)
		token_logit = self.token_mlp(x).squeeze(-1)
		token_logit = token_logit + torch.log(mask.clamp_min(1e-9))
		token_mask_all, token_idx_all = _topk_straight_through(
			token_logit, self.token_topk, dim=2, tau=self.tau_token
		)
		token_mask = token_mask_all * frame_mask.unsqueeze(-1)
		batch_arange = torch.arange(B, device=x.device)[:, None]
		x_sel_frames = x[batch_arange, frame_idx]
		token_idx = token_idx_all[batch_arange, frame_idx]
		batch_arange2 = torch.arange(B, device=x.device)[:, None, None]
		frame_arange2 = torch.arange(self.frame_topk, device=x.device)[None, :, None]
		z = x_sel_frames[batch_arange2, frame_arange2, token_idx]
		return z, frame_idx, token_idx, frame_mask, token_mask


class GraphBasedMemBank(nn.Module):
	def __init__(
		self,
		d_model: int,
		knn_k: int = 8,
		temporal_window: int = 2,
		num_layers: int = 2,
		use_gat: bool = True,
		mem_dim: Optional[int] = None,
	):
		super().__init__()
		self.knn_k = knn_k
		self.temporal_window = max(1, temporal_window)
		self.num_layers = num_layers
		self.use_gat = use_gat
		self.q_proj = nn.Linear(d_model, d_model, bias=False)
		self.k_proj = nn.Linear(d_model, d_model, bias=False)
		self.v_proj = nn.Linear(d_model, d_model, bias=False)
		self.out = nn.Linear(d_model, d_model, bias=False)
		self.norm = nn.LayerNorm(d_model)
		mem_dim = mem_dim or d_model
		self.mem_cell = nn.GRUCell(input_size=d_model, hidden_size=mem_dim)
		self.mem_to_feat = nn.Linear(mem_dim, d_model)
		if self.use_gat:
			self.att_q = nn.Linear(d_model, d_model, bias=False)
			self.att_k = nn.Linear(d_model, d_model, bias=False)
			self.att_vec = nn.Linear(d_model, 1, bias=False)
		self._mem_state: Dict[str, torch.Tensor] = {}

	@staticmethod
	def _cosine_topk(q: torch.Tensor, k: torch.Tensor, topk: int):
		q_norm = F.normalize(q, dim=-1)
		k_norm = F.normalize(k, dim=-1)
		sim = q_norm @ k_norm.t()
		vals, idx = torch.topk(sim, k=min(topk, k.shape[0]), dim=-1)
		return idx, vals

	def _edge_index_and_weights(
		self,
		z_seq: torch.Tensor,
		pos_seq: Optional[torch.Tensor] = None,
		valid_seq: Optional[torch.Tensor] = None,
	):
		T, M, D = z_seq.shape
		if valid_seq is None:
			valid_seq = torch.ones(T, M, device=z_seq.device, dtype=z_seq.dtype)
		edges = []
		for t in range(T):
			q = z_seq[t]
			q_mask = valid_seq[t]
			q_idx = torch.nonzero(q_mask > 0.5, as_tuple=False).squeeze(-1)
			if q_idx.numel() == 0:
				edges.append(([], []))
				continue
			k_list, id_map = [], []
			for dt in range(self.temporal_window + 1):
				tt = t - dt
				if tt < 0:
					break
				k_mask = valid_seq[tt]
				k_idx = torch.nonzero(k_mask > 0.5, as_tuple=False).squeeze(-1)
				if k_idx.numel() == 0:
					continue
				k_list.append(z_seq[tt][k_idx])
				id_map += [(tt, int(ii)) for ii in k_idx.tolist()]
			if len(k_list) == 0:
				edges.append(([], []))
				continue
			K = torch.cat(k_list, dim=0)
			q_feat = q[q_idx]
			nbr_idx, nbr_w = self._cosine_topk(q_feat, K, self.knn_k)
			nbr_global = []
			for row in nbr_idx.tolist():
				nbr_global.append([id_map[j] for j in row])
			edges.append(([(t, int(i.item())) for i in q_idx], (nbr_global, nbr_w)))
		return edges

	def _gat_message(self, q, K):
		qh = self.att_q(q)
		kh = self.att_k(K)
		return qh, kh

	def forward(
		self,
		z: torch.Tensor,
		pos: Optional[torch.Tensor] = None,
		valid_mask: Optional[torch.Tensor] = None,
		memory_id: Optional[str] = None,
		reset_memory: bool = False,
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		B, T, M, D = z.shape
		if valid_mask is None:
			valid_mask = torch.ones(B, T, M, device=z.device, dtype=z.dtype)
		h_list = []
		mem_key = memory_id or "default"
		if reset_memory or (mem_key not in self._mem_state):
			self._mem_state[mem_key] = torch.zeros(B, M, self.mem_cell.hidden_size, device=z.device)
		mem = self._mem_state[mem_key]
		for t in range(T):
			h_t = z[:, t]
			updated = torch.zeros_like(h_t)
			for b in range(B):
				h_seq = z[b]
				v_seq = valid_mask[b]
				edges = self._edge_index_and_weights(h_seq, pos_seq=None, valid_seq=v_seq)
				targets, (nbrs, weights) = edges[t]
				if len(targets) == 0:
					continue
				tgt_local_idx = [i for (_, i) in targets]
				q = h_seq[t][tgt_local_idx]
				msgs = []
				for row, w in zip(nbrs, weights):
					K = torch.stack([h_seq[tt][ii] for (tt, ii) in row], dim=0)
					if self.use_gat:
						qh, kh = self._gat_message(q[q.new_zeros(()).long()], K)
						att = torch.softmax((qh[0:1] * kh).sum(-1), dim=-1)
						alpha = 0.5 * w + 0.5 * att
					else:
						alpha = w
					msgs.append((alpha.unsqueeze(-1) * self.v_proj(K)).sum(dim=0))
				msg = torch.stack(msgs, dim=0)
				updated[b, tgt_local_idx] = msg
			inp = self.out(self.norm(updated + self.q_proj(h_t)))
			mem = self.mem_cell(inp.reshape(-1, D), mem.reshape(-1, self.mem_cell.hidden_size)).reshape(B, M, -1)
			h_t_out = h_t + self.mem_to_feat(mem)
			h_list.append(h_t_out)
		h = torch.stack(h_list, dim=1)
		self._mem_state[mem_key] = mem
		return h, {mem_key: mem}


# Experimental alternative modules kept for completeness -----------------

def _fps_indices_feats(X: torch.Tensor, k: int, seed_idx: int) -> torch.Tensor:
	T = X.size(0)
	k = min(k, T)
	selected = [int(seed_idx)]
	min_dist = torch.full((T,), float('inf'), device=X.device)
	for _ in range(k - 1):
		last = X[selected[-1]]
		dist = torch.norm(X - last[None, :], dim=1)
		min_dist = torch.minimum(min_dist, dist)
		min_dist[torch.tensor(selected, device=X.device)] = -1
		selected.append(int(torch.argmax(min_dist).item()))
	return torch.tensor(selected, device=X.device, dtype=torch.long)


def _fps_indices_tokens(F: torch.Tensor, k: int) -> torch.Tensor:
	N = F.size(0)
	k = min(k, N)
	mu = F.mean(dim=0, keepdim=True)
	first = int(torch.argmax(torch.norm(F - mu, dim=1)).item())
	selected = [first]
	min_dist = torch.full((N,), float('inf'), device=F.device)
	for _ in range(k - 1):
		last = F[selected[-1]]
		dist = torch.norm(F - last[None, :], dim=1)
		min_dist = torch.minimum(min_dist, dist)
		min_dist[selected] = -1
		selected.append(int(torch.argmax(min_dist).item()))
	return torch.tensor(selected, device=F.device, dtype=torch.long)


class FPSChangePointSelector(nn.Module):
	def __init__(self, d_model: int, frame_topk: int, token_topk: int, use_cls: bool = False, ema_alpha: float = 0.9):
		super().__init__()
		self.frame_topk = frame_topk
		self.token_topk = token_topk
		self.use_cls = use_cls
		self.ema_alpha = ema_alpha

	def forward(
		self,
		x: torch.Tensor,
		mask: Optional[torch.Tensor] = None,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		B, T, N, D = x.shape
		device = x.device
		if mask is None:
			mask = torch.ones(B, T, N, device=device, dtype=x.dtype)
		if self.use_cls and N >= 1:
			frame_repr = x[:, :, 0, :]
		else:
			denom = mask.sum(dim=2, keepdim=True).clamp_min(1e-6)
			frame_repr = (x * mask.unsqueeze(-1)).sum(dim=2) / denom
		frame_idx_list = []
		for b in range(B):
			fr = frame_repr[b]
			ema = fr[0].clone()
			novelty = torch.zeros(T, device=device)
			for t in range(T):
				novelty[t] = torch.norm(fr[t] - ema, p=2)
				ema = self.ema_alpha * ema + (1 - self.ema_alpha) * fr[t]
			seed = int(torch.argmax(novelty).item())
			idx_b = _fps_indices_feats(fr, self.frame_topk, seed)
			frame_idx_list.append(idx_b)
		frame_idx = torch.stack(frame_idx_list, dim=0)
		token_idx = torch.zeros(B, self.frame_topk, self.token_topk, dtype=torch.long, device=device)
		for b in range(B):
			for i in range(self.frame_topk):
				f = int(frame_idx[b, i])
				valid = mask[b, f] > 0
				F = x[b, f][valid]
				sel = _fps_indices_tokens(F, self.token_topk)
				orig_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)[sel]
				token_idx[b, i, : len(sel)] = orig_idx
		b_ar = torch.arange(B, device=device)[:, None]
		x_sel_frames = x[b_ar, frame_idx]
		b_ar2 = torch.arange(B, device=device)[:, None, None]
		fr_ar2 = torch.arange(self.frame_topk, device=device)[None, :, None]
		z = x_sel_frames[b_ar2, fr_ar2, token_idx]
		frame_mask = torch.zeros(B, T, device=device, dtype=x.dtype)
		frame_mask[b_ar, frame_idx] = 1.0
		token_mask = torch.zeros(B, T, N, device=device, dtype=x.dtype)
		token_mask[b_ar2, frame_idx[:, :, None], token_idx] = 1.0
		return z, frame_idx, token_idx, frame_mask, token_mask


class LatentCrossAttnMemBank(nn.Module):
	def __init__(self, d_model: int, num_latents: int = 64, num_heads: int = 8, ff_mult: int = 4, dropout: float = 0.0):
		super().__init__()
		self.latents = nn.Parameter(torch.randn(num_latents, d_model))
		self.enc_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
		self.dec_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
		self.ffn = nn.Sequential(
			nn.LayerNorm(d_model),
			nn.Linear(d_model, ff_mult * d_model),
			nn.GELU(),
			nn.Linear(ff_mult * d_model, d_model),
		)
		self.norm_in = nn.LayerNorm(d_model)
		self.norm_out = nn.LayerNorm(d_model)

	def forward(self, z: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
		B, T, M, D = z.shape
		x = z.reshape(B, T * M, D)
		x = self.norm_in(x)
		q = self.latents.unsqueeze(0).expand(B, -1, -1)
		lat, _ = self.enc_attn(q, x, x, key_padding_mask=None)
		y, _ = self.dec_attn(x, lat, lat)
		y = self.ffn(y) + y
		h = self.norm_out(x + y).reshape(B, T, M, D)
		return h, {"latents": lat.detach()}


class TemporalConvMemBank(nn.Module):
	def __init__(self, d_model: int, kernel_size: int = 5, dilations=(1, 2, 4), dropout: float = 0.0):
		super().__init__()
		layers: list[nn.Module] = []
		for d in dilations:
			pad = d * (kernel_size - 1) // 2
			layers += [
				nn.Conv1d(d_model, d_model, kernel_size, padding=pad, dilation=d, groups=d_model),
				nn.Conv1d(d_model, d_model, 1),
				nn.GELU(),
				nn.Dropout(dropout),
			]
		self.layers = nn.ModuleList(layers)
		self.norm = nn.LayerNorm(d_model)

	def forward(self, z: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
		B, T, M, D = z.shape
		x = z.permute(0, 2, 3, 1).reshape(B * M, D, T)
		y = x
		for i in range(0, len(self.layers), 4):
			dw = self.layers[i](y)
			pw = self.layers[i + 1](dw)
			act = self.layers[i + 2](pw)
			y = self.layers[i + 3](act) + y
		h = y.reshape(B, M, D, T).permute(0, 3, 1, 2)
		h = self.norm(h)
		return h, {}


__all__ = [
	'FrameTokenCoSelector', 'GraphBasedMemBank', 'FPSChangePointSelector',
	'LatentCrossAttnMemBank', 'TemporalConvMemBank'
]

# ---- Legacy statistical memory (moved out of main path) ----
class StatMem(nn.Module):
	def __init__(self, d_model: int, use_arp: bool = True, window: int = 8, alpha: float = 0.5):
		super().__init__()
		self.use_arp = use_arp
		self.window = max(1, int(window))
		self.alpha = float(alpha)
		self._mem_state: Dict[str, torch.Tensor] = {}

	@staticmethod
	def _arp_window(z_seq: torch.Tensor, t: int, W: int, valid_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
		device = z_seq.device
		start = max(0, t - W + 1)
		L = t - start + 1
		w = torch.arange(1, L + 1, device=device, dtype=z_seq.dtype)
		w = 2 * w - (L + 1)
		w = w / (w.abs().sum().clamp_min(1e-6))
		window = z_seq[start:t + 1]
		if valid_seq is not None:
			v = valid_seq[start:t + 1].unsqueeze(-1).to(z_seq.dtype)
			window = window * v
			denom = (v * w.view(L, 1, 1)).abs().sum(dim=0).clamp_min(1e-6)
			return (window * w.view(L, 1, 1)).sum(dim=0) / denom
		else:
			return (window * w.view(L, 1, 1)).sum(dim=0)

	def forward(self,
				z: torch.Tensor,
				pos: Optional[torch.Tensor] = None,
				valid_mask: Optional[torch.Tensor] = None,
				memory_id: Optional[str] = None,
				reset_memory: bool = False):
		B, T, M, D = z.shape
		device = z.device
		if valid_mask is None:
			valid_mask = torch.ones(B, T, M, device=device, dtype=z.dtype)
		key = memory_id or "default"
		if reset_memory or (key not in self._mem_state):
			self._mem_state[key] = torch.zeros(B, M, D, device=device, dtype=z.dtype)
		mem = self._mem_state[key]
		h_list = []
		for t in range(T):
			z_t = z[:, t]
			v_t = valid_mask[:, t].unsqueeze(-1)
			if self.use_arp:
				arp_t = torch.zeros_like(z_t)
				for b in range(B):
					arp_t[b] = self._arp_window(z[b], t, self.window, valid_seq=valid_mask[b])
				base = arp_t
			else:
				base = z_t
			h_t = (1.0 - self.alpha) * mem + self.alpha * base
			h_t = v_t * h_t + (1.0 - v_t) * mem
			h_list.append(h_t)
			mem = h_t
		h = torch.stack(h_list, dim=1)
		self._mem_state[key] = mem
		return h, {key: mem}

__all__.append('StatMem')

# ---- Lightweight statistical selector (legacy) ----
def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	a = F.normalize(a, dim=-1)
	b = F.normalize(b, dim=-1)
	return a @ b.transpose(-1, -2)


def _facility_location_greedy(frame_repr: torch.Tensor, k: int, score: Optional[torch.Tensor] = None,
							  lambda_cov: float = 1.0, lambda_score: float = 0.0,
							  valid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
	B, T, D = frame_repr.shape
	device = frame_repr.device
	sim_bt = []
	for b in range(B):
		fb = frame_repr[b]
		sim_bt.append(_safe_cosine(fb, fb))
	sim = torch.stack(sim_bt, dim=0)  # [B,T,T]

	if score is None:
		score = torch.zeros(B, T, device=device)
	if valid is None:
		valid = torch.ones(B, T, device=device)

	frame_idx = torch.zeros(B, k, dtype=torch.long, device=device)
	frame_mask = torch.zeros(B, T, dtype=frame_repr.dtype, device=device)
	best_cover = torch.zeros(B, T, device=device)
	chosen = torch.zeros(B, T, dtype=torch.bool, device=device)

	for step in range(k):
		best = best_cover.unsqueeze(-1)  # [B,T,1]
		new_best = torch.maximum(best, sim)
		cov_gain_all = (new_best - best).sum(dim=1)  # [B,T]
		total_gain = lambda_cov * cov_gain_all + lambda_score * score
		total_gain = total_gain.masked_fill(~valid.bool(), float('-inf'))
		total_gain = total_gain.masked_fill(chosen, float('-inf'))
		j = torch.argmax(total_gain, dim=1)
		frame_idx[:, step] = j
		batch_arange = torch.arange(B, device=device)
		frame_mask[batch_arange, j] = 1.0
		bj = sim[batch_arange, :, j]
		best_cover = torch.maximum(best_cover, bj)
		chosen[batch_arange, j] = True

	return frame_idx, frame_mask


def _kcenter_greedy(X: torch.Tensor, k: int, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
	N, D = X.shape
	device = X.device
	if valid is None:
		valid = torch.ones(N, device=device, dtype=torch.bool)
	else:
		valid = valid.bool()
	valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
	if valid_idx.numel() == 0:
		return X.new_zeros(k, dtype=torch.long)
	Xv = X[valid_idx]
	dists = (Xv ** 2).sum(dim=1)
	first = torch.argmax(dists)
	selected = [int(valid_idx[first].item())]
	min_dist = torch.cdist(X, X[selected].detach(), p=2).squeeze(-1)
	min_dist[~valid] = -1.0
	for _ in range(1, k):
		cand = torch.argmax(min_dist)
		selected.append(int(cand.item()))
		new_d = torch.cdist(X, X[cand].unsqueeze(0).detach(), p=2).squeeze(-1)
		min_dist = torch.minimum(min_dist, new_d)
		min_dist[~valid] = -1.0
	return torch.tensor(selected, dtype=torch.long, device=device)


class SimpleFrameTokenSelector(nn.Module):
	def __init__(self,
				 d_model: int,
				 frame_topk: int,
				 token_topk: int,
				 lambda_cov: float = 1.0,
				 lambda_score: float = 0.0,
				 use_cls: bool = False,
				 hidden: int = 4):
		super().__init__()
		self.frame_topk = frame_topk
		self.token_topk = token_topk
		self.lambda_cov = lambda_cov
		self.lambda_score = lambda_score
		self.use_cls = use_cls
		self.frame_score_head = nn.Sequential(
			nn.LayerNorm(d_model),
			nn.Linear(d_model, hidden * d_model),
			nn.GELU(),
			nn.Linear(hidden * d_model, 1),
		)

	def forward(self,
				x: torch.Tensor,
				mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		B, T, N, D = x.shape
		device = x.device
		if mask is None:
			mask = torch.ones(B, T, N, device=device, dtype=x.dtype)
		if self.use_cls and N >= 1:
			frame_repr = x[:, :, 0, :]
			valid_frame = torch.ones(B, T, device=device, dtype=torch.bool)
		else:
			denom = mask.sum(dim=2, keepdim=False).clamp_min(1e-6)[..., None]
			frame_repr = (x * mask[..., None]).sum(dim=2) / denom
			valid_frame = (denom.squeeze(-1) > 0)
		frame_score = self.frame_score_head(frame_repr).squeeze(-1)
		frame_idx, frame_mask = _facility_location_greedy(
			frame_repr, k=self.frame_topk, score=frame_score,
			lambda_cov=self.lambda_cov, lambda_score=self.lambda_score,
			valid=valid_frame.to(frame_repr.dtype)
		)
		token_idx = torch.zeros(B, self.frame_topk, self.token_topk, dtype=torch.long, device=device)
		token_mask = torch.zeros(B, T, N, dtype=x.dtype, device=device)
		batch_arange = torch.arange(B, device=device)
		for b in range(B):
			for kf in range(self.frame_topk):
				t_sel = int(frame_idx[b, kf].item())
				X = x[b, t_sel]
				m = mask[b, t_sel] > 0.5
				idxs = _kcenter_greedy(X, self.token_topk, valid=m)
				token_idx[b, kf] = idxs
				token_mask[b, t_sel, idxs] = 1.0
		x_sel_frames = x[batch_arange[:, None], frame_idx]
		z = x_sel_frames[batch_arange[:, None, None], torch.arange(self.frame_topk, device=device)[None, :, None], token_idx]
		return z, frame_idx, token_idx, frame_mask, token_mask


# ---- RamaFuse temporal fusion (legacy) ----
from math import gcd, pi


class RamaFuse(nn.Module):
	"""
	Ramanujan-based sequence fusion layer (legacy, replaced by SQuaRe-Fuse)
	Input : z:[B,T,M,D], valid_mask:[B,T,M] (optional)
	Output: h:[B,T,M,D], memory_dict (kept for API compatibility)
	"""

	def __init__(self,
				 d_model: int,
				 max_period: int = 16,
				 window: int = 16,
				 proj_dim: int = 0,
				 causal: bool = True,
				 beta_init: float = 0.5,
				 eps: float = 1e-6):
		super().__init__()
		self.Q = int(max_period)
		self.W = int(window)
		self.causal = causal
		self.eps = eps

		self.proj_dim = int(proj_dim)
		if self.proj_dim > 0:
			self.analysis_proj = nn.Linear(d_model, self.proj_dim, bias=False)
		else:
			self.analysis_proj = None

		self.gate = nn.Sequential(
			nn.Conv1d(self.Q, self.Q, kernel_size=1, groups=self.Q, bias=True),
			nn.GELU(),
			nn.Conv1d(self.Q, self.Q, kernel_size=1, bias=True),
			nn.Sigmoid()
		)

		self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
		self.register_buffer("rama_kernels", self._make_rama_kernels(self.Q, self.W), persistent=False)
		self._mem_state: Dict[str, torch.Tensor] = {}

	@staticmethod
	def _ramanujan_sum_vec(q: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
		n = torch.arange(W, device=device, dtype=dtype)
		ks = [a for a in range(1, q + 1) if gcd(a, q) == 1]
		if len(ks) == 0:
			return torch.ones(W, device=device, dtype=dtype)
		angles = torch.outer(torch.tensor(ks, device=device, dtype=dtype), n) * (2.0 * pi / q)
		c = torch.cos(angles).sum(dim=0)
		c = c - c.mean()
		denom = torch.sqrt(torch.clamp(torch.sum(c * c), min=1e-6))
		c = c / denom
		return c

	def _make_rama_kernels(self, Q: int, W: int) -> torch.Tensor:
		ker = []
		for q in range(1, Q + 1):
			ker.append(self._ramanujan_sum_vec(q, W, device=torch.device("cpu"), dtype=torch.float32))
		K = torch.stack(ker, dim=0).unsqueeze(1)
		return K

	def _pad(self, x: torch.Tensor) -> torch.Tensor:
		if self.causal and self.W > 1:
			return F.pad(x, (self.W - 1, 0))
		else:
			pad = (self.W - 1) // 2
			return F.pad(x, (pad, self.W - 1 - pad))

	def forward(self,
				z: torch.Tensor,
				pos: torch.Tensor = None,
				valid_mask: torch.Tensor = None,
				memory_id: str = None,
				reset_memory: bool = False):
		B, T, M, D = z.shape
		device, dtype = z.device, z.dtype
		if valid_mask is None:
			valid_mask = torch.ones(B, T, M, device=device, dtype=dtype)

		if self.analysis_proj is None:
			z_anl = z.mean(dim=-1)
		else:
			z_flat = z.view(B * T * M, D)
			proj = self.analysis_proj(z_flat).view(B, T, M, self.proj_dim)
			z_anl = proj.mean(dim=-1)

		z_anl = z_anl * valid_mask
		x = z_anl.permute(0, 2, 1).contiguous()
		x = x.view(B * M, 1, T)
		x = self._pad(x)

		rama_k = self.rama_kernels.to(device=device, dtype=dtype)
		r = F.conv1d(x, rama_k, bias=None, stride=1, padding=0)
		r = r.view(B, M, self.Q, T).permute(0, 2, 1, 3).contiguous()

		g = self.gate(r.view(B * M, self.Q, T))
		g = g.view(B, M, self.Q, T).permute(0, 2, 1, 3).contiguous()

		z_syn = (z * valid_mask.unsqueeze(-1)).permute(0, 2, 3, 1).contiguous()
		y = z_syn.view(B * M * D, 1, T)
		y = self._pad(y)
		R = F.conv1d(y, rama_k, bias=None, stride=1, padding=0)
		R = R.view(B, M, D, self.Q, T).permute(0, 3, 1, 2, 4).contiguous()

		p = (R * g.unsqueeze(3)).sum(dim=1)
		p = p.permute(0, 3, 1, 2).contiguous()

		h = z + self.beta * p
		h = valid_mask.unsqueeze(-1) * h + (1.0 - valid_mask.unsqueeze(-1)) * z

		key = memory_id or "default"
		if reset_memory or (key not in self._mem_state):
			self._mem_state[key] = torch.zeros(B, M, D, device=device, dtype=dtype)
		return h, {key: self._mem_state[key]}


__all__ += ['SimpleFrameTokenSelector', 'RamaFuse']

