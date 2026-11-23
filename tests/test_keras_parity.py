import numpy as np
import pytest
import torch

import model as pt_mod
from keras_model import build_keras_model, copy_weights_from_pytorch


def _get_pytorch_model_cls():
    return next(
        obj for obj in vars(pt_mod).values()
        if isinstance(obj, type) and issubclass(obj, torch.nn.Module)
    )


@pytest.mark.parametrize("hidden", [4, 16, 32, 64])
def test_parameter_count_parity(hidden: int) -> None:
    n_features = 9
    pt_cls = _get_pytorch_model_cls()
    pt_net = pt_cls(n_features=n_features, hidden=hidden)
    k_net = build_keras_model(n_features=n_features, hidden=hidden)

    pt_params = sum(p.numel() for p in pt_net.parameters())
    k_params = k_net.count_params()
    assert pt_params == k_params


def test_copied_weights_forward_parity() -> None:
    torch.manual_seed(2026)
    n_features = 9
    hidden = 16

    pt_cls = _get_pytorch_model_cls()
    pt_net = pt_cls(n_features=n_features, hidden=hidden).eval()
    k_net = copy_weights_from_pytorch(build_keras_model(n_features=n_features, hidden=hidden), pt_net)

    rng = np.random.default_rng(2026)
    x_np = rng.standard_normal((12, n_features)).astype(np.float32)

    pt_out = pt_net(torch.from_numpy(x_np)).detach().cpu().numpy()
    k_out = np.asarray(k_net(x_np, training=False))

    assert pt_out.shape == k_out.shape == (12,)
    np.testing.assert_allclose(k_out, pt_out, rtol=1e-5, atol=1e-5)
