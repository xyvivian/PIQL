# from trainer_embedder.gmm_test import *
# from trainer_embedder.embedder  import *


from trainer_embedder.gmm_test import *
from trainer_embedder.embedder  import *

import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


# =========================
# 1. Parser / tokenizer
# =========================

def _maybe_number(x: str):
    x = x.strip()
    if x == "":
        return x
    if re.fullmatch(r"[+-]?\d+", x):
        return int(x)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", x):
        return float(x)
    return x


def _parse_csv_payload(payload: str) -> List[str]:
    parts = [p.strip() for p in payload.split(",")]
    return [p for p in parts if p and p != "..."]


def tokenize_program(
    text: str,
    num_cls: int = 10,
    max_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert bracket-format string into mixed typed tokens.
    """
    chunks = re.findall(r"\[([^\[\]]+)\]", text)

    tokens: List[Dict[str, Any]] = []
    current_entity_type: Optional[str] = None
    current_entity_id: Optional[int] = None
    current_block: Optional[str] = None
    current_family: Optional[str] = None

    if num_cls < 1:
        raise ValueError(f"num_cls must be >= 1, got {num_cls}")

    tokens.extend({"type": "symbol", "name": "CLS"} for _ in range(num_cls))

    for chunk in chunks:
        chunk = chunk.strip()

        if ":" not in chunk:
            name = chunk
            tokens.append({"type": "symbol", "name": name})

            if name == "ENTITY":
                current_block = "ENTITY"
                current_entity_type = None
                current_entity_id = None
            elif name == "OUTLIER":
                current_block = "OUTLIER"
                current_entity_type = None
                current_entity_id = None
            else:
                current_block = name
            continue

        key, raw_value = chunk.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        tokens.append({"type": "symbol", "name": key})

        if key == "FAMILY":
            current_family = raw_value
            tokens.append({"type": "symbol", "name": raw_value})
            continue

        if key == "TYPE":
            tokens.append({"type": "symbol", "name": raw_value})
            if current_block == "ENTITY":
                current_entity_type = raw_value
            continue

        if key in {"DIM", "N_COMP", "ID", "THR_MINUS", "THR_PLUS"}:
            val = _maybe_number(raw_value)
            if not isinstance(val, (int, float)):
                raise ValueError(f"Expected numeric value for {key}, got {raw_value!r}")
            tok = {
                "type": "scalar",
                "field": key,
                "value": float(val),
                "family": current_family,
                "block": current_block,
            }
            if current_entity_type is not None:
                tok["entity_type"] = current_entity_type
            if current_entity_id is not None:
                tok["entity_id"] = current_entity_id

            tokens.append(tok)

            if key == "ID":
                current_entity_id = int(val)
                tokens[-1]["entity_id"] = int(val)
            continue

        if key == "COPULA_PARAM":
            tokens.append({"type": "symbol", "name": raw_value})
            continue

        if key in {"SUB_DIMS", "PARENTS"}:
            values = _parse_csv_payload(raw_value)
            for pos, x in enumerate(values):
                v = _maybe_number(x)
                if not isinstance(v, (int, float)):
                    continue
                tokens.append(
                    {
                        "type": "index_entry",
                        "field": key,
                        "position": pos,
                        "value": int(round(float(v))),
                        "family": current_family,
                        "block": current_block,
                        "entity_type": current_entity_type,
                        "entity_id": current_entity_id,
                    }
                )
            continue

        if key in {"MEAN", "COV_DIAG", "INF_COV", "COEFFS", "PARAMS", "NOISE_PARAMS"}:
            values = _parse_csv_payload(raw_value)
            for dim, x in enumerate(values):
                v = _maybe_number(x)
                if not isinstance(v, (int, float)):
                    continue
                tokens.append(
                    {
                        "type": "vector_entry",
                        "field": key,
                        "dim": dim,
                        "value": float(v),
                        "family": current_family,
                        "block": current_block,
                        "entity_type": current_entity_type,
                        "entity_id": current_entity_id,
                    }
                )
            continue

        # fallback: scalar / vector / symbol
        parts = _parse_csv_payload(raw_value)
        parsed = [_maybe_number(p) for p in parts]

        if len(parsed) == 1 and isinstance(parsed[0], (int, float)):
            tokens.append(
                {
                    "type": "scalar",
                    "field": key,
                    "value": float(parsed[0]),
                    "family": current_family,
                    "block": current_block,
                    "entity_type": current_entity_type,
                    "entity_id": current_entity_id,
                }
            )
        elif len(parsed) > 1 and all(isinstance(v, (int, float)) for v in parsed):
            for dim, v in enumerate(parsed):
                tokens.append(
                    {
                        "type": "vector_entry",
                        "field": key,
                        "dim": dim,
                        "value": float(v),
                        "family": current_family,
                        "block": current_block,
                        "entity_type": current_entity_type,
                        "entity_id": current_entity_id,
                    }
                )
        else:
            tokens.append({"type": "symbol", "name": raw_value})

    tokens.append({"type": "symbol", "name": "SEP"})

    if max_tokens is not None:
        if max_tokens < 2:
            raise ValueError(f"max_tokens must be >= 2, got {max_tokens}")
        if len(tokens) > max_tokens:
            tokens = tokens[: max_tokens - 1] + [{"type": "symbol", "name": "SEP"}]

    return tokens


# =========================
# 2. Build vocabularies
# =========================

def build_vocab(tokens_list: List[List[Dict[str, Any]]], key: str) -> Dict[str, int]:
    vocab = {"[PAD]": 0}
    for tokens in tokens_list:
        for tok in tokens:
            if key in tok and tok[key] is not None:
                val = str(tok[key])
                if val not in vocab:
                    vocab[val] = len(vocab)
    return vocab


# =========================
# 3. Dataset-to-vectors module
# =========================

class ProgramVectorizer(nn.Module):
    """
    Converts parsed mixed tokens into learnable vectors suitable for a Transformer.
    """
    def __init__(
        self,
        symbol_vocab: Dict[str, int],
        field_vocab: Dict[str, int],
        family_vocab: Dict[str, int],
        entity_type_vocab: Dict[str, int],
        d_model: int = 128,
        max_dim: int = 512,
        max_entity_id: int = 64,
        max_index_pos: int = 512,
    ):
        super().__init__()
        self.symbol_vocab = symbol_vocab
        self.field_vocab = field_vocab
        self.family_vocab = family_vocab
        self.entity_type_vocab = entity_type_vocab

        self.d_model = d_model

        self.symbol_emb = nn.Embedding(len(symbol_vocab), d_model)
        self.field_emb = nn.Embedding(len(field_vocab), d_model)
        self.family_emb = nn.Embedding(len(family_vocab), d_model)
        self.entity_type_emb = nn.Embedding(len(entity_type_vocab), d_model)

        self.entity_id_emb = nn.Embedding(max_entity_id, d_model)
        self.dim_emb = nn.Embedding(max_dim, d_model)
        self.index_pos_emb = nn.Embedding(max_index_pos, d_model)

        self.value_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.layernorm = nn.LayerNorm(d_model)

    @staticmethod
    def normalize_value(x: float) -> float:
        # signed log transform
        x_t = torch.tensor(float(x), dtype=torch.float32)
        return torch.sign(x_t) * torch.log1p(torch.abs(x_t))

    def encode_numeric_value(self, x: float, device=None) -> torch.Tensor:
        x_norm = self.normalize_value(x).view(1, 1)
        if device is not None:
            x_norm = x_norm.to(device)
        return self.value_mlp(x_norm).squeeze(0).squeeze(0)

    def _lookup(self, vocab: Dict[str, int], value: Optional[str]) -> int:
        if value is None:
            return 0
        return vocab.get(str(value), 0)

    def embed_token(self, tok: Dict[str, Any], device=None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device

        out = torch.zeros(self.d_model, device=device)

        tok_type = tok["type"]

        if tok_type == "symbol":
            idx = self._lookup(self.symbol_vocab, tok["name"])
            out = out + self.symbol_emb(torch.tensor(idx, device=device))

        elif tok_type == "scalar":
            field_idx = self._lookup(self.field_vocab, tok["field"])
            out = out + self.field_emb(torch.tensor(field_idx, device=device))
            out = out + self.encode_numeric_value(tok["value"], device=device)

            family_idx = self._lookup(self.family_vocab, tok.get("family"))
            out = out + self.family_emb(torch.tensor(family_idx, device=device))

            ent_type_idx = self._lookup(self.entity_type_vocab, tok.get("entity_type"))
            out = out + self.entity_type_emb(torch.tensor(ent_type_idx, device=device))

            entity_id = int(tok.get("entity_id", 0) or 0)
            entity_id = max(0, min(entity_id, self.entity_id_emb.num_embeddings - 1))
            out = out + self.entity_id_emb(torch.tensor(entity_id, device=device))

        elif tok_type == "vector_entry":
            field_idx = self._lookup(self.field_vocab, tok["field"])
            out = out + self.field_emb(torch.tensor(field_idx, device=device))
            out = out + self.encode_numeric_value(tok["value"], device=device)

            family_idx = self._lookup(self.family_vocab, tok.get("family"))
            out = out + self.family_emb(torch.tensor(family_idx, device=device))

            ent_type_idx = self._lookup(self.entity_type_vocab, tok.get("entity_type"))
            out = out + self.entity_type_emb(torch.tensor(ent_type_idx, device=device))

            entity_id = int(tok.get("entity_id", 0) or 0)
            entity_id = max(0, min(entity_id, self.entity_id_emb.num_embeddings - 1))
            out = out + self.entity_id_emb(torch.tensor(entity_id, device=device))

            dim = int(tok["dim"])
            dim = max(0, min(dim, self.dim_emb.num_embeddings - 1))
            out = out + self.dim_emb(torch.tensor(dim, device=device))

        elif tok_type == "index_entry":
            field_idx = self._lookup(self.field_vocab, tok["field"])
            out = out + self.field_emb(torch.tensor(field_idx, device=device))

            family_idx = self._lookup(self.family_vocab, tok.get("family"))
            out = out + self.family_emb(torch.tensor(family_idx, device=device))

            ent_type_idx = self._lookup(self.entity_type_vocab, tok.get("entity_type"))
            out = out + self.entity_type_emb(torch.tensor(ent_type_idx, device=device))

            entity_id = int(tok.get("entity_id", 0) or 0)
            entity_id = max(0, min(entity_id, self.entity_id_emb.num_embeddings - 1))
            out = out + self.entity_id_emb(torch.tensor(entity_id, device=device))

            pos = int(tok.get("position", 0))
            pos = max(0, min(pos, self.index_pos_emb.num_embeddings - 1))
            out = out + self.index_pos_emb(torch.tensor(pos, device=device))

            dim_idx = int(tok["value"])
            dim_idx = max(0, min(dim_idx, self.dim_emb.num_embeddings - 1))
            out = out + self.dim_emb(torch.tensor(dim_idx, device=device))

        else:
            raise ValueError(f"Unknown token type: {tok_type}")

        return self.layernorm(out)

    def forward(self, tokens: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Returns:
            x: [seq_len, d_model]
        """
        device = next(self.parameters()).device
        vecs = [self.embed_token(tok, device=device) for tok in tokens]
        return torch.stack(vecs, dim=0)


# =========================
# 4. Transformer encoder
# =========================

class ProgramTransformerEncoder(nn.Module):
    def __init__(
        self,
        vectorizer: ProgramVectorizer,
        num_cls: int = 10,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.vectorizer = vectorizer
        self.num_cls = num_cls
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Distinct learnable embeddings per CLS slot.
        self.cls_token_emb = nn.Embedding(num_cls, d_model)
        # Positional embeddings break permutation symmetry across sequence positions.
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, tokens: List[Dict[str, Any]]) -> torch.Tensor:
        """
        tokens -> program embedding
        Returns:
            cls_repr: [num_cls * d_model]
        """
        x = self.vectorizer(tokens)      # [seq_len, d_model]
        x = x.unsqueeze(0)               # [1, seq_len, d_model]

        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos_ids)

        cls_count = min(self.num_cls, seq_len)
        if cls_count > 0:
            cls_ids = torch.arange(cls_count, device=x.device)
            x[:, :cls_count, :] = x[:, :cls_count, :] + self.cls_token_emb(cls_ids).unsqueeze(0)

        h = self.encoder(x)              # [1, seq_len, d_model]
        cls_repr = h[:, :cls_count, :].reshape(-1)  # [cls_count * d_model]
        return cls_repr

    def encode_sequence(self, tokens: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Returns the full encoded sequence.
        """
        x = self.vectorizer(tokens).unsqueeze(0)
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos_ids)

        cls_count = min(self.num_cls, seq_len)
        if cls_count > 0:
            cls_ids = torch.arange(cls_count, device=x.device)
            x[:, :cls_count, :] = x[:, :cls_count, :] + self.cls_token_emb(cls_ids).unsqueeze(0)

        h = self.encoder(x)
        return h.squeeze(0)              # [seq_len, d_model]






if __name__ == "__main__":
    text = """[FAMILY:GMM] [DIM:27] [N_COMP:3]
[ENTITY] [TYPE:COMPONENT] [ID:0] [MEAN:-3.483,0.008,-2.007,-0.515,0.673,0.237,0.965,0.908,3.098,-3.650,...] [COV_DIAG:2.466,1.462,2.541,1.344,0.243,1.780,1.850,0.918,0.144,0.155,...]
[ENTITY] [TYPE:COMPONENT] [ID:1] [MEAN:0.000,-1.446,-1.426,0.647,-2.643,-1.071,0.244,0.137,-0.088,0.108,...] [COV_DIAG:1.257,0.167,0.912,1.384,3.420,1.699,1.042,1.789,0.216,3.041,...]
[ENTITY] [TYPE:COMPONENT] [ID:2] [MEAN:-0.317,0.456,0.000,0.000,-0.399,-0.042,1.256,-2.368,0.727,0.463,...] [COV_DIAG:3.146,1.130,0.486,0.047,0.756,2.332,0.735,2.588,3.366,2.587,...]
[OUTLIER] [TYPE:inflated_cov] [SUB_DIMS:3.000,4.000,5.000,6.000,9.000,10.000,12.000,14.000,21.000]
[ENTITY] [TYPE:OUTLIER_COMPONENT] [ID:0] [INF_COV:2.466,1.462,2.541,6.721,1.217,8.898,9.252,0.918,0.144,0.774,...]
[ENTITY] [TYPE:OUTLIER_COMPONENT] [ID:1] [INF_COV:1.257,0.167,0.912,6.918,17.098,8.493,5.212,1.789,0.216,15.203,...]
[ENTITY] [TYPE:OUTLIER_COMPONENT] [ID:2] [INF_COV:3.146,1.130,0.486,0.237,3.778,11.661,3.677,2.588,3.366,12.933,...]"""

    tokens = tokenize_program(text)

    print("First 20 parsed tokens:")
    for t in tokens[:20]:
        print(t)

    # Build vocabs from one or more tokenized programs
    all_tokens = [tokens]

    symbol_vocab = build_vocab(all_tokens, "name")
    field_vocab = build_vocab(all_tokens, "field")
    family_vocab = build_vocab(all_tokens, "family")
    entity_type_vocab = build_vocab(all_tokens, "entity_type")

    print("\nSymbol vocab:", symbol_vocab)
    print("Field vocab:", field_vocab)
    print("Family vocab:", family_vocab)
    print("Entity type vocab:", entity_type_vocab)

    vectorizer = ProgramVectorizer(
        symbol_vocab=symbol_vocab,
        field_vocab=field_vocab,
        family_vocab=family_vocab,
        entity_type_vocab=entity_type_vocab,
        d_model=128,
        max_dim=256,
        max_entity_id=32,
        max_index_pos=256,
    )

    model = ProgramTransformerEncoder(
        vectorizer=vectorizer,
        num_cls=10,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
    )

    # Convert string to sequence of vectors
    seq_vectors = vectorizer(tokens)  # [seq_len, d_model]
    print("\nSequence vectors shape:", seq_vectors.shape)
    print(seq_vectors)

    # Encode with transformer
    prog_embedding = model(tokens)    # [d_model]
    print("Program embedding shape:", prog_embedding.shape)

    full_hidden = model.encode_sequence(tokens)  # [seq_len, d_model]
    print("Encoded hidden states shape:", full_hidden.shape)