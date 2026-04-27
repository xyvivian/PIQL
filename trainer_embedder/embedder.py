import torch


def _fmt_scalar(x, decimals=3):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().item()
    return f"{float(x):.{decimals}f}"


def _fmt_vec(x, decimals=3, max_len=10):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().reshape(-1)
        vals = x.tolist()
    else:
        vals = list(x)

    shown = vals[:max_len]
    body = ",".join(
        f"{float(v):.{decimals}f}" if decimals > 0 else f"{int(v)}"
        for v in shown
    )
    if len(vals) > max_len:
        body += ",..."
    return body


def _extract_value_for_text(v, decimals=3, max_len=10):
    """
    Generic formatter:
      - scalar -> string
      - 1D tensor/list -> compact vector string
      - 2D tensor -> diagonal if square else flattened head
      - str/int -> string
    """
    if isinstance(v, str):
        return v

    if isinstance(v, (int, float)):
        if isinstance(v, int):
            return str(v)
        return f"{float(v):.{decimals}f}"

    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
        if v.ndim == 0:
            return f"{float(v.item()):.{decimals}f}"
        if v.ndim == 1:
            return _fmt_vec(v, decimals=decimals, max_len=max_len)
        if v.ndim == 2:
            if v.shape[0] == v.shape[1]:
                diag = torch.diagonal(v, dim1=-2, dim2=-1)
                return _fmt_vec(diag, decimals=decimals, max_len=max_len)
            return _fmt_vec(v.reshape(-1), decimals=decimals, max_len=max_len)

    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return ""
        if isinstance(v[0], str):
            shown = v[:max_len]
            body = ",".join(str(x) for x in shown)
            if len(v) > max_len:
                body += ",..."
            return body
        return _fmt_vec(v, decimals=decimals, max_len=max_len)

    return str(v)


def _format_tag_block(tag, fields, decimals=3, max_len_map=None):
    """
    Example:
      tag='ENTITY'
      fields={'TYPE': 'COMPONENT', 'ID': 0, 'WEIGHT': 0.5}
    ->
      [ENTITY] [TYPE:COMPONENT] [ID:0] [WEIGHT:0.500]
    """
    if max_len_map is None:
        max_len_map = {}

    parts = [f"[{tag}]"]
    for key, value in fields.items():
        max_len = max_len_map.get(key, 10)
        value_txt = _extract_value_for_text(value, decimals=decimals, max_len=max_len)
        parts.append(f"[{key}:{value_txt}]")
    return " ".join(parts)


def describe_prior_program(
    family,
    global_fields,
    entities=None,
    blocks=None,
    decimals=3,
    max_len_map=None,
):
    """
    Generic serializer for program-like priors.

    Args:
        family: str, e.g. 'GMM', 'SCM', 'COPULA'
        global_fields: dict of top-level fields
        entities: list[dict], each item like
            {
                "tag": "ENTITY",
                "fields": {"TYPE": "COMPONENT", "ID": 0, "WEIGHT": 0.3, ...}
            }
        blocks: list[dict], same structure as entities, for extra sections
        decimals: float formatting precision
        max_len_map: optional dict mapping field name -> truncation length

    Returns:
        str
    """
    if entities is None:
        entities = []
    if blocks is None:
        blocks = []
    if max_len_map is None:
        max_len_map = {}

    lines = []
    header_fields = {"FAMILY": family, **global_fields}
    lines.append(
        " ".join(
            f"[{k}:{_extract_value_for_text(v, decimals=decimals, max_len=max_len_map.get(k, 10))}]"
            for k, v in header_fields.items()
        )
    )

    for item in entities:
        tag = item.get("tag", "ENTITY")
        fields = item["fields"]
        lines.append(
            _format_tag_block(
                tag=tag,
                fields=fields,
                decimals=decimals,
                max_len_map=max_len_map,
            )
        )

    for item in blocks:
        tag = item.get("tag", "BLOCK")
        fields = item["fields"]
        lines.append(
            _format_tag_block(
                tag=tag,
                fields=fields,
                decimals=decimals,
                max_len_map=max_len_map,
            )
        )

    return "\n".join(lines)