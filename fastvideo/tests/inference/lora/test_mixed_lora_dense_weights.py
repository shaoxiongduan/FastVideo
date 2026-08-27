"""Unit coverage for dense weights carried beside LoRA factors."""

import torch
import torch.nn as nn

from fastvideo.pipelines.lora_pipeline import LoRAPipeline


class _Attention(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.to_gate_compress = nn.Linear(3, 4, bias=False)
        self.projection = nn.Linear(3, 4, bias=True)


class _Block(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()


class _Transformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block()])


def _identity_mapping(name: str) -> tuple[str, None, None]:
    return name, None, None


def test_full_adapter_weight_is_loaded_and_removed_before_lora_parsing() -> None:
    transformer = _Transformer()
    dense = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    dense_bias = torch.arange(4, dtype=torch.float32)
    adapter = {
        "transformer_blocks.0.attn.to_gate_compress.weight": dense.clone(),
        "transformer_blocks.0.attn.projection.bias": dense_bias.clone(),
        "transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 3),
    }

    count = LoRAPipeline._load_full_adapter_weights(transformer, adapter, _identity_mapping)

    assert count == 2
    torch.testing.assert_close(transformer.transformer_blocks[0].attn.to_gate_compress.weight, dense)
    torch.testing.assert_close(transformer.transformer_blocks[0].attn.projection.bias, dense_bias)
    assert "transformer_blocks.0.attn.to_gate_compress.weight" not in adapter
    assert "transformer_blocks.0.attn.projection.bias" not in adapter
    assert "transformer_blocks.0.attn.to_q.lora_A.weight" in adapter
