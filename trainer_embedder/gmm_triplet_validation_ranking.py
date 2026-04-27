import argparse
import random
from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch

from lightning_gmm_triplet_train import (
    LitGMMMetricLearner,
    ProgramTransformerEncoder,
    ProgramVectorizer,
    build_bootstrap_vocabs,
    make_positive_from_base,
    params_to_description,
    sample_base_gmm_params,
)


@dataclass
class EvalStats:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    triplet_acc: float
    mean_pos_dist: float
    mean_neg_dist: float


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


def _embed_text(model: LitGMMMetricLearner, text: str) -> torch.Tensor:
    with torch.no_grad():
        return model._encode_text(text).detach().cpu().reshape(-1)


def run_ranking_eval(model: LitGMMMetricLearner, args: argparse.Namespace) -> EvalStats:
    recalls_1 = 0
    recalls_5 = 0
    recalls_10 = 0
    rr_sum = 0.0

    triplet_correct = 0
    pos_dists = []
    neg_dists = []

    for _ in range(args.eval_queries):
        dim = int(np.random.randint(args.eval_min_dim, args.eval_max_dim + 1))
        num_cluster = int(np.random.randint(args.eval_min_cluster, args.eval_max_cluster + 1))
        max_mean = int(np.random.randint(args.eval_min_max_mean, args.eval_max_max_mean + 1))
        max_var = int(np.random.randint(args.eval_min_max_var, args.eval_max_max_var + 1))
        inflate_scale = float(np.random.uniform(args.eval_min_inflate, args.eval_max_inflate))

        anchor = sample_base_gmm_params(
            dim=dim,
            num_cluster=num_cluster,
            max_mean=max_mean,
            max_var=max_var,
            inflate_scale=inflate_scale,
            device=torch.device("cpu"),
        )
        positive = make_positive_from_base(
            anchor,
            inflate_scale=inflate_scale,
            mean_noise_std=args.mean_noise_std,
            diag_noise_std=args.diag_noise_std,
            inf_noise_std=args.inf_noise_std,
        )

        a_txt = params_to_description(anchor, device=torch.device("cpu"))
        p_txt = params_to_description(positive, device=torch.device("cpu"))

        a_emb = _embed_text(model, a_txt)
        p_emb = _embed_text(model, p_txt)

        candidates = [(p_emb, 1)]
        for _ in range(args.eval_candidates - 1):
            neg_max_mean = int(np.random.randint(1, max(2, max_mean * 3) + 1))
            neg_max_var = int(np.random.randint(1, max(2, max_var * 3) + 1))
            neg_inflate_scale = float(np.random.uniform(1.5, max(2.0, inflate_scale * 3.0)))

            negative = sample_base_gmm_params(
                dim=dim,
                num_cluster=num_cluster,
                max_mean=neg_max_mean,
                max_var=neg_max_var,
                inflate_scale=neg_inflate_scale,
                device=torch.device("cpu"),
            )
            n_txt = params_to_description(negative, device=torch.device("cpu"))
            n_emb = _embed_text(model, n_txt)
            candidates.append((n_emb, 0))

        d_pos = torch.norm(a_emb - p_emb, p=2).item()
        all_neg_d = [torch.norm(a_emb - emb, p=2).item() for emb, y in candidates if y == 0]
        mean_neg = float(np.mean(all_neg_d))
        pos_dists.append(d_pos)
        neg_dists.append(mean_neg)

        if d_pos + args.margin < min(all_neg_d):
            triplet_correct += 1

        dist_labels = []
        for emb, label in candidates:
            d = torch.norm(a_emb - emb, p=2).item()
            dist_labels.append((d, label))
        dist_labels.sort(key=lambda x: x[0])

        positive_rank = next(i + 1 for i, (_, lbl) in enumerate(dist_labels) if lbl == 1)
        recalls_1 += int(positive_rank <= 1)
        recalls_5 += int(positive_rank <= 5)
        recalls_10 += int(positive_rank <= 10)
        rr_sum += 1.0 / positive_rank

    n = float(args.eval_queries)
    return EvalStats(
        recall_at_1=recalls_1 / n,
        recall_at_5=recalls_5 / n,
        recall_at_10=recalls_10 / n,
        mrr=rr_sum / n,
        triplet_acc=triplet_correct / n,
        mean_pos_dist=float(np.mean(pos_dists)),
        mean_neg_dist=float(np.mean(neg_dists)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation/ranking pipeline for gmm_triplet checkpoints.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/ocean/projects/cis250290p/xding/FoMo-Meta_0413/trainer_embedder/ckpt/gmm_triplet/gmm_triplet.d27.k3.cls10.E100.step50.bs16.lr0.001.nneg8.20260420_000803/seed42/epoch-epoch=08.ckpt",
    )
    parser.add_argument("--seed", type=int, default=42)

    # model args
    parser.add_argument("--dim", type=int, default=27)
    parser.add_argument("--num_cluster", type=int, default=3)
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

    # positive perturbation strength (same meaning as training)
    parser.add_argument("--mean_noise_std", type=float, default=0.1)
    parser.add_argument("--diag_noise_std", type=float, default=0.1)
    parser.add_argument("--inf_noise_std", type=float, default=0.1)

    # eval regime
    parser.add_argument("--eval_queries", type=int, default=400)
    parser.add_argument("--eval_candidates", type=int, default=64)
    parser.add_argument("--eval_min_dim", type=int, default=6)
    parser.add_argument("--eval_max_dim", type=int, default=40)
    parser.add_argument("--eval_min_cluster", type=int, default=2)
    parser.add_argument("--eval_max_cluster", type=int, default=8)
    parser.add_argument("--eval_min_max_mean", type=int, default=3)
    parser.add_argument("--eval_max_max_mean", type=int, default=12)
    parser.add_argument("--eval_min_max_var", type=int, default=2)
    parser.add_argument("--eval_max_max_var", type=int, default=10)
    parser.add_argument("--eval_min_inflate", type=float, default=2.0)
    parser.add_argument("--eval_max_inflate", type=float, default=10.0)

    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    model.eval()

    stats = run_ranking_eval(model, args)
    print("=== Ranking Validation ===")
    print(f"Recall@1:   {stats.recall_at_1:.4f}")
    print(f"Recall@5:   {stats.recall_at_5:.4f}")
    print(f"Recall@10:  {stats.recall_at_10:.4f}")
    print(f"MRR:        {stats.mrr:.4f}")
    print("=== Margin/Distance Validation ===")
    print(f"Triplet Acc (margin={args.margin}): {stats.triplet_acc:.4f}")
    print(f"Mean pos dist: {stats.mean_pos_dist:.4f}")
    print(f"Mean neg dist: {stats.mean_neg_dist:.4f}")
    print(f"Gap (neg-pos): {stats.mean_neg_dist - stats.mean_pos_dist:.4f}")


if __name__ == "__main__":
    main()
