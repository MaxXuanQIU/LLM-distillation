import torch
import torch.nn.functional as F
import pytest
from kd.loss import forward_kl, response_loss
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
    from trl import GKDTrainer
    s, t = torch.randn(2, 4, 17), torch.randn(2, 4, 17)
    expected = GKDTrainer.generalized_jsd_loss(s, t, beta=0.0, reduction="sum")
    torch.testing.assert_close(forward_kl(s, t), expected)


@pytest.mark.parametrize("soft", [False, True])
def test_chunked_loss_and_all_parameter_gradients_match_dense(soft):
    from copy import deepcopy
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=19, hidden_size=16, intermediate_size=32,
                     num_hidden_layers=1, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=8, attention_dropout=0.)
    torch.manual_seed(10)
    student = Qwen3ForCausalLM(cfg)
    dense = deepcopy(student)
    teacher = Qwen3ForCausalLM(cfg).eval().requires_grad_(False) if soft else None
    ids, mask, labels = collate([{"prompt_ids": [1, 2, 3]}, {"prompt_ids": [4]}],
                               [[5, 9], [6, 7, 9]], 0, "cpu")
    loss, n = response_loss(student, teacher, ids, mask, labels, chunk_size=2)
    selected = labels[:, 1:] != -100
    sl = dense(input_ids=ids, attention_mask=mask).logits[:, :-1][selected]
    if soft:
        with torch.no_grad():
            tl = teacher(input_ids=ids, attention_mask=mask).logits[:, :-1][selected]
        reference = forward_kl(sl, tl)
    else:
        reference = F.cross_entropy(sl, labels[:, 1:][selected], reduction="sum")
    assert n == 5
    torch.testing.assert_close(loss, reference)
    loss.backward()
    reference.backward()
    for param, expected in zip(student.parameters(), dense.parameters()):
        torch.testing.assert_close(param.grad, expected.grad, atol=2e-6, rtol=2e-5)
