from collections import OrderedDict
from typing import Any, Dict, List, Tuple
import math
import time

import numpy as np
import tenseal as ts
import torch


class CKKSContextManager:
    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] | None = None,
        global_scale_power: int = 40,
    ) -> None:
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]

        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale_power = global_scale_power
        self.context = self._build_context()

    def _build_context(self) -> ts.Context:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
        )
        context.global_scale = 2 ** self.global_scale_power
        context.generate_galois_keys()
        context.generate_relin_keys()
        return context

    def serialize_context(self) -> bytes:
        return self.context.serialize(save_secret_key=True)


class TensorPacker:
    @staticmethod
    def state_dict_to_flat_vector(
        state_dict: "OrderedDict[str, torch.Tensor]",
    ) -> Tuple[np.ndarray, List[Tuple[str, torch.Size, int, int]]]:
        flat_parts: List[np.ndarray] = []
        metadata: List[Tuple[str, torch.Size, int, int]] = []
        cursor = 0

        for name, tensor in state_dict.items():
            array = tensor.detach().cpu().float().numpy().reshape(-1)
            length = array.size
            flat_parts.append(array)
            metadata.append((name, tensor.shape, cursor, cursor + length))
            cursor += length

        if not flat_parts:
            raise ValueError("Empty state_dict cannot be packed.")

        flat_vector = np.concatenate(flat_parts, axis=0).astype(np.float64)
        return flat_vector, metadata

    @staticmethod
    def flat_vector_to_state_dict(
        flat_vector: np.ndarray,
        metadata: List[Tuple[str, torch.Size, int, int]],
        reference_state_dict: "OrderedDict[str, torch.Tensor]",
        device: torch.device,
    ) -> "OrderedDict[str, torch.Tensor]":
        restored = OrderedDict()

        for name, shape, start, end in metadata:
            values = flat_vector[start:end]
            tensor = torch.tensor(values, dtype=reference_state_dict[name].dtype)
            tensor = tensor.view(shape).to(device)
            restored[name] = tensor

        return restored

    @staticmethod
    def split_vector(vector: np.ndarray, chunk_size: int) -> List[np.ndarray]:
        chunks: List[np.ndarray] = []
        total = len(vector)

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunks.append(vector[start:end])

        return chunks

    @staticmethod
    def merge_chunks(chunks: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks, axis=0)


class CKKSUpdateEncryptor:
    def __init__(self, context: ts.Context, chunk_size: int = 4096) -> None:
        self.context = context
        self.chunk_size = chunk_size

    def encrypt_vector(
        self, vector: np.ndarray
    ) -> Tuple[List[ts.CKKSVector], Dict[str, Any]]:
        start_time = time.time()

        chunks = TensorPacker.split_vector(vector, self.chunk_size)
        encrypted_chunks = [ts.ckks_vector(self.context, chunk.tolist()) for chunk in chunks]

        elapsed = time.time() - start_time

        serialized_bytes = 0
        for enc in encrypted_chunks:
            serialized_bytes += len(enc.serialize())

        stats = {
            "num_chunks": len(encrypted_chunks),
            "plaintext_values": int(len(vector)),
            "serialized_ciphertext_bytes": int(serialized_bytes),
            "encrypt_time_sec": float(elapsed),
        }
        return encrypted_chunks, stats

    def decrypt_vector(
        self, encrypted_chunks: List[ts.CKKSVector]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.time()

        chunks: List[np.ndarray] = []
        for enc in encrypted_chunks:
            dec = np.array(enc.decrypt(), dtype=np.float64)
            chunks.append(dec)

        vector = TensorPacker.merge_chunks(chunks)
        elapsed = time.time() - start_time

        stats = {
            "decrypt_time_sec": float(elapsed),
            "num_chunks": len(encrypted_chunks),
            "plaintext_values": int(len(vector)),
        }
        return vector, stats


class EncryptedAggregator:
    @staticmethod
    def weighted_average(
        encrypted_updates: List[List[ts.CKKSVector]],
        weights: List[float],
    ) -> List[ts.CKKSVector]:
        if len(encrypted_updates) == 0:
            raise ValueError("No encrypted updates provided.")

        if len(encrypted_updates) != len(weights):
            raise ValueError("Number of encrypted updates and weights must match.")

        num_chunks = len(encrypted_updates[0])

        for update in encrypted_updates:
            if len(update) != num_chunks:
                raise ValueError("All encrypted updates must have the same number of chunks.")

        total_weight = float(sum(weights))
        if total_weight <= 0:
            raise ValueError("Total aggregation weight must be positive.")

        normalized_weights = [w / total_weight for w in weights]

        aggregated: List[ts.CKKSVector] = []
        for chunk_idx in range(num_chunks):
            acc = encrypted_updates[0][chunk_idx] * normalized_weights[0]
            for client_idx in range(1, len(encrypted_updates)):
                acc += encrypted_updates[client_idx][chunk_idx] * normalized_weights[client_idx]
            aggregated.append(acc)

        return aggregated


def compute_model_update(
    local_state: "OrderedDict[str, torch.Tensor]",
    global_state: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    update = OrderedDict()

    for name in global_state.keys():
        update[name] = local_state[name].detach().cpu() - global_state[name].detach().cpu()

    return update


def apply_model_update(
    global_state: "OrderedDict[str, torch.Tensor]",
    update_state: "OrderedDict[str, torch.Tensor]",
    device: torch.device,
) -> "OrderedDict[str, torch.Tensor]":
    new_state = OrderedDict()

    for name in global_state.keys():
        new_state[name] = (
            global_state[name].detach().cpu() + update_state[name].detach().cpu()
        ).to(device)

    return new_state


def l2_norm_of_state_dict(state_dict: "OrderedDict[str, torch.Tensor]") -> float:
    total = 0.0

    for tensor in state_dict.values():
        total += float(torch.sum(tensor.detach().cpu().float() ** 2).item())

    return math.sqrt(total)