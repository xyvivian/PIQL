import argparse
import os
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.manifold import TSNE

from lightning_gmm_triplet_train import (
	LitGMMMetricLearner,
	ProgramTransformerEncoder,
	ProgramVectorizer,
	build_bootstrap_vocabs,
	params_to_description,
	sample_base_gmm_params,
)


def build_model(
	*,
	dim: int,
	num_cluster: int,
	num_cls: int,
	max_tokens: int | None,
	d_model: int,
	nhead: int,
	num_layers: int,
	dim_feedforward: int,
	dropout: float,
	lr: float,
	margin: float,
	warmup_ratio: float,
	min_lr_ratio: float,
) -> LitGMMMetricLearner:
	symbol_vocab, field_vocab, family_vocab, entity_type_vocab = build_bootstrap_vocabs(
		num_bootstrap=64,
		num_cls=num_cls,
		dim=dim,
		num_cluster=num_cluster,
		max_tokens=max_tokens,
	)

	vectorizer = ProgramVectorizer(
		symbol_vocab=symbol_vocab,
		field_vocab=field_vocab,
		family_vocab=family_vocab,
		entity_type_vocab=entity_type_vocab,
		d_model=d_model,
		max_dim=max(256, dim + 32),
		max_entity_id=max(32, num_cluster + 8),
		max_index_pos=max(256, dim + 32),
	)

	encoder = ProgramTransformerEncoder(
		vectorizer=vectorizer,
		num_cls=num_cls,
		d_model=d_model,
		nhead=nhead,
		num_layers=num_layers,
		dim_feedforward=dim_feedforward,
		dropout=dropout,
	)
	return LitGMMMetricLearner(
		encoder=encoder,
		num_cls=num_cls,
		max_tokens=max_tokens,
		lr=lr,
		margin=margin,
		warmup_ratio=warmup_ratio,
		min_lr_ratio=min_lr_ratio,
	)


def load_weights(model: LitGMMMetricLearner, ckpt_path: str) -> None:
	ckpt = torch.load(ckpt_path, map_location="cpu")
	state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
	result = model.load_state_dict(state_dict, strict=False)
	if getattr(result, "missing_keys", None):
		print("[ckpt] missing_keys:", result.missing_keys)
	if getattr(result, "unexpected_keys", None):
		print("[ckpt] unexpected_keys:", result.unexpected_keys)


def main() -> None:
	parser = argparse.ArgumentParser(description="Load GMM triplet checkpoint and visualize embeddings with t-SNE.")
	parser.add_argument(
		"--ckpt",
		type=str,
		default="/ocean/projects/cis250290p/xding/FoMo-Meta_0413/trainer_embedder/ckpt/gmm_triplet/gmm_triplet.d27.k3.cls10.E100.step50.bs16.lr0.001.nneg8.20260420_000803/seed42/epoch-epoch=08.ckpt",
	)
	parser.add_argument("--seed", type=int, default=42)

	# Model args (must match training)
	parser.add_argument("--num_cls", type=int, default=10)
	parser.add_argument("--max_tokens", type=int, default=None)
	parser.add_argument("--d_model", type=int, default=256)
	parser.add_argument("--nhead", type=int, default=8)
	parser.add_argument("--num_layers", type=int, default=4)
	parser.add_argument("--dim_feedforward", type=int, default=256)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--margin", type=float, default=1.0)
	parser.add_argument("--warmup_ratio", type=float, default=0.1)
	parser.add_argument("--min_lr_ratio", type=float, default=0.0)

	# GMM sampling ranges
	parser.add_argument("--n_samples", type=int, default=220)
	parser.add_argument("--min_dim", type=int, default=6)
	parser.add_argument("--max_dim", type=int, default=40)
	parser.add_argument("--min_cluster", type=int, default=2)
	parser.add_argument("--max_cluster", type=int, default=8)
	parser.add_argument("--min_max_mean", type=int, default=3)
	parser.add_argument("--max_max_mean", type=int, default=12)
	parser.add_argument("--min_max_var", type=int, default=2)
	parser.add_argument("--max_max_var", type=int, default=10)
	parser.add_argument("--min_inflate", type=float, default=2.0)
	parser.add_argument("--max_inflate", type=float, default=10.0)

	# t-SNE and output
	parser.add_argument("--perplexity", type=float, default=30.0)
	parser.add_argument("--output", type=str, default="/ocean/projects/cis250290p/xding/FoMo-Meta_0413/trainer_embedder/gmm_tsne_epoch08.png")
	args = parser.parse_args()

	pl.seed_everything(args.seed, workers=True)
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	if not os.path.exists(args.ckpt):
		raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# Use representative bootstrap settings for vocab construction.
	model = build_model(
		dim=27,
		num_cluster=3,
		num_cls=args.num_cls,
		max_tokens=args.max_tokens,
		d_model=args.d_model,
		nhead=args.nhead,
		num_layers=args.num_layers,
		dim_feedforward=args.dim_feedforward,
		dropout=args.dropout,
		lr=args.lr,
		margin=args.margin,
		warmup_ratio=args.warmup_ratio,
		min_lr_ratio=args.min_lr_ratio,
	)
	load_weights(model, args.ckpt)
	model.eval().to(device)

	embeds: List[np.ndarray] = []
	dims: List[int] = []
	clusters: List[int] = []
	max_means: List[int] = []
	max_vars: List[int] = []
	inflates: List[float] = []

	with torch.no_grad():
		for _ in range(args.n_samples):
			dim = int(np.random.randint(args.min_dim, args.max_dim + 1))
			num_cluster = int(np.random.randint(args.min_cluster, args.max_cluster + 1))
			max_mean = int(np.random.randint(args.min_max_mean, args.max_max_mean + 1))
			max_var = int(np.random.randint(args.min_max_var, args.max_max_var + 1))
			inflate_scale = float(np.random.uniform(args.min_inflate, args.max_inflate))

			gmm_params = sample_base_gmm_params(
				dim=dim,
				num_cluster=num_cluster,
				max_mean=max_mean,
				max_var=max_var,
				inflate_scale=inflate_scale,
				device=torch.device("cpu"),
			)
			gmm_text = params_to_description(gmm_params, device=torch.device("cpu"))
			emb_raw = model._encode_text(gmm_text).detach().cpu().reshape(10,256).numpy()[0]
			#print("emb shape:", emb_raw.shape)
			emb = emb_raw

			embeds.append(emb)
			dims.append(dim)
			clusters.append(num_cluster)
			max_means.append(max_mean)
			max_vars.append(max_var)
			inflates.append(inflate_scale)

	X = np.stack(embeds, axis=0)
	tsne = TSNE(
		n_components=2,
		perplexity=min(args.perplexity, max(5.0, (args.n_samples - 1) / 3.0)),
		init="pca",
		learning_rate="auto",
		random_state=args.seed,
	)
	Z = tsne.fit_transform(X)

	fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
	axes = axes.ravel()

	panels = [
		(np.array(dims), "dim"),
		(np.array(clusters), "num_cluster"),
		(np.array(max_means), "max_mean"),
		(np.array(max_vars), "max_var"),
		(np.array(inflates), "inflate_scale"),
	]

	for i, (vals, title) in enumerate(panels):
		sc = axes[i].scatter(Z[:, 0], Z[:, 1], c=vals, cmap="viridis", s=18, alpha=0.9)
		axes[i].set_title(f"t-SNE colored by {title}")
		axes[i].set_xlabel("t-SNE 1")
		axes[i].set_ylabel("t-SNE 2")
		cbar = fig.colorbar(sc, ax=axes[i], fraction=0.046, pad=0.04)
		cbar.set_label(title)

	axes[-1].axis("off")
	axes[-1].text(
		0.02,
		0.85,
		f"n_samples={args.n_samples}\nnum_cls={args.num_cls}\nd_model={args.d_model}",
		fontsize=9,
		transform=axes[-1].transAxes,
	)

	out_dir = os.path.dirname(args.output)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
	fig.savefig(args.output, dpi=220)
	print(f"Saved t-SNE figure to: {args.output}")


if __name__ == "__main__":
	main()

