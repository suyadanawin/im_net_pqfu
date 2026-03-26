from typing import Dict, List

import pandas as pd
import torch


def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    vectors = []
    for key in state_dict.keys():
        vectors.append(state_dict[key].detach().float().cpu().reshape(-1))
    return torch.cat(vectors)


def model_l2_distance(state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor]) -> float:
    vec_a = flatten_state_dict(state_a)
    vec_b = flatten_state_dict(state_b)
    return torch.norm(vec_a - vec_b, p=2).item()


def model_update_norm(prev_state: Dict[str, torch.Tensor], next_state: Dict[str, torch.Tensor]) -> float:
    return model_l2_distance(prev_state, next_state)


def per_layer_l2_distance(state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor]) -> List[Dict]:
    rows = []
    for key in state_a.keys():
        tensor_a = state_a[key].detach().float().cpu()
        tensor_b = state_b[key].detach().float().cpu()
        dist = torch.norm((tensor_a - tensor_b).reshape(-1), p=2).item()
        rows.append({
            "layer_name": key,
            "l2_distance": dist,
            "num_params": tensor_a.numel()
        })
    return rows


def last_layer_distance(state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor]) -> Dict:
    out = {}
    for key in ["fc.weight", "fc.bias"]:
        if key in state_a and key in state_b:
            dist = torch.norm(
                (state_a[key].detach().float().cpu() - state_b[key].detach().float().cpu()).reshape(-1),
                p=2
            ).item()
            out[f"{key.replace('.', '_')}_l2_distance"] = dist
        else:
            out[f"{key.replace('.', '_')}_l2_distance"] = None
    return out


def add_round_column(rows: List[Dict], round_idx: int, tag: str) -> List[Dict]:
    out = []
    for row in rows:
        row2 = dict(row)
        row2["round"] = round_idx
        row2["tag"] = tag
        out.append(row2)
    return out


def save_rows_csv(rows: List[Dict], path: str):
    pd.DataFrame(rows).to_csv(path, index=False)