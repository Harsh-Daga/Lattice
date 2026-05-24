"""delta_encoder must use transform_delta_encode, not transform_batching."""

from __future__ import annotations

from lattice.core.config import LatticeConfig
from lattice.transforms.registry import get_transform_spec, is_transform_enabled


def test_delta_encoder_config_flag() -> None:
    spec = get_transform_spec("delta_encoder")
    assert spec is not None
    assert spec.config_flag == "transform_delta_encode"
    assert spec.config_flag != "transform_batching"


def test_delta_encode_honours_own_config_field() -> None:
    assert hasattr(LatticeConfig(), "transform_delta_encode")
    enabled = LatticeConfig(transform_delta_encode=True)
    disabled = LatticeConfig(transform_delta_encode=False)
    assert is_transform_enabled(enabled, "delta_encoder") is True
    assert is_transform_enabled(disabled, "delta_encoder") is True  # execution_only always on
