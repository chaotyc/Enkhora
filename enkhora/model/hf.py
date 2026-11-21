"""HF adapter: swap each sparse MoE block for a RemoteMoE that calls an executor.

Supported: transformers v5 Qwen3-MoE and OLMoE (fused ``experts.gate_up_proj`` /
``experts.down_proj`` and a ``gate`` router returning (logits, scores, indices)).
The trunk (embeddings, attention, norms, router gates, lm_head) stays on the anchor;
the expert parameters are exported to the master store and removed.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn

from ..experts import BundleWeights, ExpertSpec, split_experts
from ..store import MasterStore
from ..wire import dtype_name


class LayerExecutor(Protocol):
    def run_layer(self, layer: int, hidden: torch.Tensor, topk_idx: torch.Tensor,
                  topk_w: torch.Tensor) -> torch.Tensor: ...


class RemoteMoE(nn.Module):
    """Drop-in replacement for a sparse MoE block. Keeps the router on the anchor."""

    def __init__(self, layer: int, gate: nn.Module, executor: LayerExecutor | None = None):
        super().__init__()
        self.layer = layer
        self.gate = gate              # same attribute name keeps state_dict keys and router hooks
        self.executor = executor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        _, weights, indices = self.gate(flat)
        out = self.executor.run_layer(self.layer, flat, indices, weights)
        return out.reshape(shape)


def _layers(model: nn.Module):
    base = getattr(model, "model", model)
    return base.layers


def find_moe_blocks(model: nn.Module) -> list[tuple[int, nn.Module]]:
    out = []
    for i, layer in enumerate(_layers(model)):
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, RemoteMoE):
            out.append((i, mlp))
        elif mlp is not None and hasattr(mlp, "gate") and hasattr(getattr(mlp, "experts", None), "gate_up_proj"):
            out.append((i, mlp))
    return out


def spec_from_model(model: nn.Module) -> ExpertSpec:
    cfg = model.config
    _, block = find_moe_blocks(model)[0]
    dtype = block.experts.gate_up_proj.dtype
    inter = getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size
    return ExpertSpec(hidden=cfg.hidden_size, inter=inter, act=cfg.hidden_act, dtype=dtype_name(dtype))


def export_master(model: nn.Module, root: str, bundle_size: int, model_info: dict | None = None) -> MasterStore:
    """Write every MoE layer's experts to the master store, bundle by bundle.

    Bundles are produced lazily, so peak extra RAM is one bundle, not a second copy
    of the expert tier."""
    blocks = find_moe_blocks(model)
    if not blocks:
        raise ValueError("no supported sparse MoE blocks found")
    spec = spec_from_model(model)
    num_experts = blocks[0][1].experts.gate_up_proj.shape[0]

    def bundles():
        bid = 0
        for layer, block in blocks:
            gu, dn = block.experts.gate_up_proj.data, block.experts.down_proj.data
            for exps in split_experts(num_experts, bundle_size):
                idx = torch.tensor(exps)
                yield BundleWeights(bid, layer, exps, gu[idx].clone().cpu(), dn[idx].clone().cpu())
                bid += 1

    return MasterStore.write(root, spec, num_experts, bundle_size, bundles(), model_info)


def patch_model(model: nn.Module, executor: LayerExecutor | None = None) -> nn.Module:
    """Replace sparse MoE blocks with RemoteMoE in place and drop expert weights."""
    layers = _layers(model)
    for i, block in find_moe_blocks(model):
        if isinstance(block, RemoteMoE):
            block.executor = executor
            continue
        layers[i].mlp = RemoteMoE(i, block.gate, executor)
        del block.experts
    return model


def set_executor(model: nn.Module, executor: LayerExecutor):
    for _, block in find_moe_blocks(model):
        block.executor = executor


def tiny_qwen3_moe(seed: int = 0, std: float = 0.08, **overrides) -> nn.Module:
    """A small random-weight Qwen3-MoE with the real architecture. Deterministic in ``seed``."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    kw = dict(vocab_size=256, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
              num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
              num_experts=32, num_experts_per_tok=4, norm_topk_prob=True,
              max_position_embeddings=512, initializer_range=std, tie_word_embeddings=False,
              pad_token_id=0, bos_token_id=1, eos_token_id=None)
    kw.update(overrides)
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**kw))
    # Router weights with a larger scale so routing is decisive, not near-uniform.
    g = torch.Generator().manual_seed(seed + 1)
    for _, block in find_moe_blocks(model):
        w = block.gate.weight
        w.data = torch.randn(w.shape, generator=g) / kw["hidden_size"] ** 0.5 * 4
    return model.eval()


def load_model(name: str, seed: int = 0, dtype: str | None = None, device: str = "cpu") -> nn.Module:
    if name == "tiny":
        return tiny_qwen3_moe(seed=seed).to(device)
    from transformers import AutoModelForCausalLM

    kw = {"dtype": getattr(torch, dtype)} if dtype else {}
    # NOTE v1 limitation: this materializes the experts once before patching drops them.
    return AutoModelForCausalLM.from_pretrained(name, **kw).to(device).eval()


@torch.no_grad()
def generate(model: nn.Module, prompts: list[list[int]], max_new_tokens: int, pad_id: int = 0):
    """Greedy batched decode with HF's KV cache. Returns (tokens[B, new], logits[steps, B, V])."""
    device = next(model.parameters()).device
    L = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), L), dtype=torch.long)
    for i, p in enumerate(prompts):          # left padding
        ids[i, L - len(p):] = torch.tensor(p)
        mask[i, L - len(p):] = 1
    out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), do_sample=False,
                         max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                         pad_token_id=pad_id, output_logits=True, return_dict_in_generate=True)
    return out.sequences[:, L:].cpu(), torch.stack(out.logits).cpu()
