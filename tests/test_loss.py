import torch
import torch.nn.functional as F
from kd.loss import forward_kl
from kd.common import collate, trim_completion


def test_forward_kl_direction_and_gradient():
    s = torch.tensor([[2., -1., .5], [0., 1., -2.]], requires_grad=True)
    t = torch.tensor([[-2., 1., 0.], [1., -.5, 2.]])
    loss = forward_kl(s, t)
    p, q = t.softmax(-1), s.softmax(-1)
    expected = (p * (p.log() - q.log())).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(s.grad, q - p)
    assert not torch.isclose(loss, forward_kl(t, s.detach()))


def test_equal_distributions_zero_loss():
    x = torch.randn(4, 23)
    torch.testing.assert_close(forward_kl(x, x), torch.tensor(0.), atol=1e-6, rtol=0)


def test_response_mask_causal_shift_and_eos():
    rows = [{"prompt_ids": [1, 2, 3]}, {"prompt_ids": [4]}]
    ids, attn, labels = collate(rows, [[5, 9], [6, 7, 9]], 9, "cpu")
    assert labels[:, 1:][labels[:, 1:] != -100].tolist() == [5, 9, 6, 7, 9]
    assert attn.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    assert labels[1, -1].item() == -100
    assert trim_completion([6, 9, 9], 9) == [6, 9]


def test_trl_forward_kl_parity():
    from trl.experimental.gkd import GKDTrainer
    s, t = torch.randn(2, 4, 17), torch.randn(2, 4, 17)
    expected = GKDTrainer.generalized_jsd_loss(s, t, beta=0.0, reduction="sum")
    torch.testing.assert_close(forward_kl(s, t), expected)
