from collections import OrderedDict
from typing import List, Dict

import torch


def fedavg_aggregate(client_results: List[Dict]) -> OrderedDict:
    """
    Weighted FedAvg aggregation by number of client samples.
    """
    total_samples = sum(result["num_samples"] for result in client_results)
    if total_samples == 0:
        raise ValueError("Total client samples is 0. Cannot aggregate.")

    aggregated_state = OrderedDict()

    first_state = client_results[0]["state_dict"]
    for key in first_state.keys():
        aggregated_state[key] = torch.zeros_like(first_state[key], dtype=torch.float32)

    for result in client_results:
        client_weight = result["num_samples"] / total_samples
        client_state = result["state_dict"]

        for key in aggregated_state.keys():
            aggregated_state[key] += client_state[key].float() * client_weight

    return aggregated_state