import pytest

torch = pytest.importorskip("torch")

from urbiva_ai.models.uva_net import UVANetS10, build_model, count_parameters  # noqa: E402


def test_parameter_count_is_within_budget():
    m = build_model(width_multiplier=1.0)
    assert count_parameters(m) == 3_963_191
    assert count_parameters(m) <= 10_000_000


def test_budget_is_enforced():
    with pytest.raises(ValueError, match="limit"):
        build_model(width_multiplier=1.75)


def test_forward_shape():
    m = UVANetS10().eval()
    with torch.no_grad():
        y = m(torch.zeros(2, 3, 320, 320))
    assert y.shape == (2, 7)


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    m = UVANetS10(width_multiplier=0.5)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randn(8, 3, 96, 96)
    y = torch.randint(0, 7, (8,))
    loss_fn = torch.nn.CrossEntropyLoss()
    first = None
    for _ in range(15):
        opt.zero_grad()
        loss = loss_fn(m(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < first
