"""AddressedLinear — nn.Linear where weight rows are addressed prototypes.

Each output row of the weight matrix gets an HAddr:
  face, sub_face  — deterministic hash of the layer name
  zeck            — Zeckendorf decomposition of the row index

Rows at the same HAddr share the SAME nn.Parameter, giving geometric
weight-tying for free. The prototypes live in an AddressedParamStore so
they persist across runs — block weights warm-start instead of reiniting
from random.

Usage:
    store = AddressedParamStore()
    layer = AddressedLinear(64, 256, name="block0.w1", store=store)
    y = layer(x)

    # At checkpoint save time — EMA sync trained rows back to store:
    layer.sync_to_store(store, alpha=0.1)

    # At next run startup — prototypes already in store, AddressedLinear
    # will init from them automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from hierarchical_address import HAddr, zeckendorf
from addressed_param_store import AddressedParamStore


def layer_haddr(layer_name: str, row_idx: int) -> HAddr:
    """Stable HAddr for weight row (layer_name, row_idx).

    face / sub_face: structural address derived from layer name hash.
    zeck:            row position in Fibonacci basis.

    Two layers with the same name get the same face/sub_face, so their
    rows are co-addressed by row index — explicit inter-layer weight tying.
    Two layers with different names land on different faces, keeping them
    geometrically distinct.
    """
    h = hash(layer_name) & 0xFFFFFFFF
    face = h % 12
    sub_face = (h >> 4) % 3
    zeck = zeckendorf(row_idx) if row_idx > 0 else frozenset()
    return HAddr(face=face, sub_face=sub_face, zeck=zeck)


class AddressedLinear(nn.Module):
    """Drop-in replacement for nn.Linear with addressed, persistent weights.

    Weight rows are grouped by HAddr. Rows that share an address share one
    nn.Parameter — one Adam state entry, one gradient accumulation, one
    optimizer step. For d=64 output rows that cluster into N unique
    addresses, the optimizer processes N updates instead of 64.
    """

    def __init__(self, in_features: int, out_features: int,
                 name: str, store: AddressedParamStore,
                 bias: bool = False, n_buckets: int = 0, bucket_mode: str = "mod"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        # n_buckets: if >0 and < out_features, map row i → bucket(i) before addressing,
        # so many rows COLLIDE onto one address → share one prototype. This is the
        # parameters-as-addresses lever: unique params = n_buckets·in vs out·in.
        # bucket_mode: "mod" = i % n_buckets (uniform, naive). "phi" = Fibonacci-tier
        # folding — substrate-structured sharing (groups grow by golden ratio). Tests
        # whether the φ-substrate's own structure makes a BETTER sharing pattern.
        self.n_buckets = n_buckets
        self.bucket_mode = bucket_mode
        _FIB = [1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597]
        def _row_addr_idx(i):
            if not (n_buckets and n_buckets < out_features):
                return i
            if bucket_mode == "phi":
                # fold i into a Fibonacci-tier index, then wrap to n_buckets
                tier = 0
                for t, f in enumerate(_FIB):
                    if i < f:
                        tier = t; break
                else:
                    tier = len(_FIB)
                return tier % n_buckets
            return i % n_buckets   # "mod"
        self._row_addr_idx = _row_addr_idx

        # Map addr_key → nn.Parameter (one per unique address)
        addr_to_data: Dict[str, torch.Tensor] = {}
        self._row_keys: List[str] = []  # addr_key per output row

        for i in range(out_features):
            ai = _row_addr_idx(i)
            addr = layer_haddr(name, ai)
            wkey = f"{name}_row{ai}"
            key = AddressedParamStore._addr_key(addr)

            if key not in addr_to_data:
                prior = store.get_weight_prototype(wkey)
                if prior is not None and prior.shape == (in_features,):
                    data = prior.clone()
                else:
                    # Kaiming uniform init — same as nn.Linear default
                    data = torch.empty(in_features)
                    nn.init.kaiming_uniform_(data.unsqueeze(0),
                                            a=5 ** 0.5).squeeze_(0)
                addr_to_data[key] = data
            self._row_keys.append(key)

        # ParameterDict keys can't contain '.' — sanitize
        self._safe_keys = {k: k.replace('.', '__dot__').replace('-', '__dash__')
                           for k in addr_to_data}
        self._unsafe_keys = {v: k for k, v in self._safe_keys.items()}
        self.prototypes = nn.ParameterDict(
            {self._safe_keys[k]: nn.Parameter(v)
             for k, v in addr_to_data.items()}
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Register all rows in the store (links wkey → addr_key for EMA sync)
        for i in range(out_features):
            ai = self._row_addr_idx(i)
            addr = layer_haddr(name, ai)
            wkey = f"{name}_row{ai}"
            key = self._row_keys[i]
            store.register_weight(wkey, addr,
                                  self.prototypes[self._safe_keys[key]].data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = [self.prototypes[self._safe_keys[k]] for k in self._row_keys]
        weight = torch.stack(rows, dim=0)  # [out_features, in_features]
        return F.linear(x, weight, self.bias)

    def sync_to_store(self, store: AddressedParamStore, alpha: float = 0.1) -> None:
        """EMA sync all trained row prototypes back into the store."""
        for i in range(self.out_features):
            wkey = f"{self.name}_row{i}"
            key = self._row_keys[i]
            vec = self.prototypes[self._safe_keys[key]].detach().cpu()
            store.update_weight(wkey, vec, alpha=alpha)

    def n_unique_addresses(self) -> int:
        return len(set(self._row_keys))

    def compression_ratio(self) -> float:
        return self.out_features / max(1, self.n_unique_addresses())

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"name={self.name!r}, "
                f"unique_addrs={self.n_unique_addresses()} "
                f"(ratio={self.compression_ratio():.1f}x)")
