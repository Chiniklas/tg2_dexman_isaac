from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexsafedagger_lab.distillation.utils.loss_utils import clip_gradients_and_step


def test_finite_gradients_update_parameters() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = model.weight.detach().clone()

    model(torch.ones((1, 2))).sum().backward()
    stepped, grad_norm = clip_gradients_and_step(model.parameters(), optimizer, 1.0)

    assert stepped
    assert grad_norm == pytest.approx(2.0**0.5)
    assert not torch.equal(model.weight.detach(), before)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_gradients_do_not_update_parameters(bad_value: float) -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, bad_value)

    stepped, grad_norm = clip_gradients_and_step(model.parameters(), optimizer, 1.0)

    assert not stepped
    assert not torch.isfinite(torch.tensor(grad_norm))
    assert torch.equal(model.weight.detach(), before)
    assert model.weight.grad is None
    assert not optimizer.state
