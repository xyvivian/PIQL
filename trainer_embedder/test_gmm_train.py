import argparse
import random

import numpy as np
import pytorch_lightning as pl
import torch

from lightning_gmm_triplet_train import (
	LitGMMMetricLearner,
	ProgramTransformerEncoder,
	ProgramVectorizer,
	build_bootstrap_vocabs,
	params_to_description,
	sample_base_gmm_params,
	tokenize_program,
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
	parser = argparse.ArgumentParser(description="Load a trained GMM text embedder checkpoint and get one embedding.")
	parser.add_argument("--ckpt", type=str, default="/ocean/projects/cis250290p/xding/FoMo-Meta_0413/training_embedder/ckpt/gmm_triplet/gmm_triplet.d27.k3.cls10.E100.step50.bs16.lr0.001.nneg8.20260420_000803/seed42/epoch-epoch=09.ckpt")
	parser.add_argument("--seed", type=int, default=42)

	# Keep defaults aligned with lightning_gmm_triplet_train.py
	parser.add_argument("--dim", type=int, default=27)
	parser.add_argument("--num_cluster", type=int, default=3)
	parser.add_argument("--max_mean", type=int, default=6)
	parser.add_argument("--max_var", type=int, default=6)
	parser.add_argument("--inflate_scale", type=float, default=5.0)

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

	args = parser.parse_args()

	pl.seed_everything(args.seed, workers=True)
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = build_model(
		dim=args.dim,
		num_cluster=args.num_cluster,
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

	# Build one random GMM, convert to text description, then embed.
	gmm_params = sample_base_gmm_params(
		dim=args.dim,
		num_cluster=args.num_cluster,
		max_mean=args.max_mean,
		max_var=args.max_var,
		inflate_scale=args.inflate_scale,
		device=torch.device("cpu"),
	)
	gmm_text = params_to_description(gmm_params, device=torch.device("cpu"))

	with torch.no_grad():
		embedding = model._encode_text(gmm_text)
	#print(embedding.detach().cpu())

	num_cls_tokens = args.num_cls
	total_vals = embedding.numel()
	if total_vals % num_cls_tokens != 0:
		raise ValueError(
			f"Cannot reshape embedding of numel={total_vals} into (num_cls={num_cls_tokens}, embed_dim)."
		)
	embed_dim = total_vals // num_cls_tokens
	embedding = embedding.reshape(num_cls_tokens, embed_dim)


	print("=== GMM Description ===")
	print(gmm_text)
	print("\n=== Embedding ===")
	print("shape:", tuple(embedding.shape))
	print(embedding.detach().cpu())


if __name__ == "__main__":
	main()
