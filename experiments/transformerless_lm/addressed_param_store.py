"""AddressedParamStore — parameter slots keyed by hierarchical address.

Each HAddr owns a [d] prototype vector.  Chunk tokens promoted to the
same address inherit that prototype as their initialization seed, giving
the model geometric weight-tying for free — no explicit budget required.

The address space (dodecahedron face × sub_face × Zeckendorf integer)
is infinite; the number of distinct parameter slots grows only as new
addresses are visited.  Two chunks at the same address are
geometrically equivalent from the model's perspective.
"""

import torch
from typing import Dict, FrozenSet, List, Optional

from hierarchical_address import HAddr


class AddressedParamStore:

    def __init__(self) -> None:
        self._prototypes: Dict[str, torch.Tensor] = {}
        self._token_to_key: Dict[int, str] = {}
        self._key_to_tokens: Dict[str, List[int]] = {}
        # String-keyed weights (e.g. linear layer rows) share the same
        # prototype pool but are indexed separately from token integers.
        self._weight_to_key: Dict[str, str] = {}

    @staticmethod
    def _addr_key(addr: HAddr) -> str:
        return f"f{addr.face}_s{addr.sub_face}_z{'_'.join(map(str, sorted(addr.zeck)))}"

    def register_chunk(self, token_idx: int, addr: HAddr, seed_embed: torch.Tensor) -> torch.Tensor:
        key = self._addr_key(addr)

        if key not in self._prototypes:
            self._prototypes[key] = seed_embed.detach().cpu().float().clone()
            self._key_to_tokens[key] = []

        self._token_to_key[token_idx] = key
        if token_idx not in self._key_to_tokens[key]:
            self._key_to_tokens[key].append(token_idx)

        return self._prototypes[key]

    def update_prototype(self, token_idx: int, new_vec: torch.Tensor, alpha: float = 0.1) -> None:
        key = self._token_to_key.get(token_idx)
        if key is None:
            return
        proto = self._prototypes[key]
        v = new_vec.detach().cpu().float()
        # EMA across all siblings — proto drifts toward the centroid of trained embeddings
        self._prototypes[key] = (1.0 - alpha) * proto + alpha * v

    def get_siblings(self, token_idx: int) -> List[int]:
        key = self._token_to_key.get(token_idx)
        if key is None:
            return []
        return [t for t in self._key_to_tokens[key] if t != token_idx]

    def register_weight(self, weight_key: str, addr: HAddr,
                        seed: torch.Tensor) -> torch.Tensor:
        """Register a named weight row (e.g. a linear layer row) in the store.

        Returns the prototype (may differ from seed if address already exists).
        """
        key = self._addr_key(addr)
        if key not in self._prototypes:
            self._prototypes[key] = seed.detach().cpu().float().clone()
            self._key_to_tokens[key] = []
        self._weight_to_key[weight_key] = key
        return self._prototypes[key].clone()

    def get_weight_prototype(self, weight_key: str) -> Optional[torch.Tensor]:
        key = self._weight_to_key.get(weight_key)
        if key is None:
            return None
        proto = self._prototypes.get(key)
        return proto.clone() if proto is not None else None

    def update_weight(self, weight_key: str, new_vec: torch.Tensor,
                      alpha: float = 0.1) -> None:
        key = self._weight_to_key.get(weight_key)
        if key is None:
            return
        proto = self._prototypes[key]
        v = new_vec.detach().cpu().float()
        self._prototypes[key] = (1.0 - alpha) * proto + alpha * v

    def get_prototype(self, addr: HAddr) -> Optional[torch.Tensor]:
        key = self._addr_key(addr)
        proto = self._prototypes.get(key)
        return proto.clone() if proto is not None else None

    def n_addresses(self) -> int:
        return len(self._prototypes)

    def n_registered(self) -> int:
        return len(self._token_to_key)

    def summary(self) -> str:
        shared = sum(1 for tokens in self._key_to_tokens.values() if len(tokens) > 1)
        return (
            f"AddressedParamStore: {self.n_addresses()} addresses, "
            f"{self.n_registered()} tokens, "
            f"{shared} shared_addrs"
        )

    def to_dict(self) -> dict:
        return {
            "prototypes": {k: v.tolist() for k, v in self._prototypes.items()},
            "token_to_key": self._token_to_key,
            "key_to_tokens": self._key_to_tokens,
            "weight_to_key": self._weight_to_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AddressedParamStore":
        store = cls()
        store._prototypes = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in d["prototypes"].items()
        }
        store._token_to_key = {int(k): v for k, v in d["token_to_key"].items()}
        store._key_to_tokens = {
            k: [int(t) for t in v] for k, v in d["key_to_tokens"].items()
        }
        store._weight_to_key = dict(d.get("weight_to_key", {}))
        return store
