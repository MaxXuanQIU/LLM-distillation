from copy import deepcopy

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from kd.common import collate
from kd.loss import forward_kl, response_loss
from studies.followup.measure import rotations


def test_equal_answer_weights_match_dense_loss_and_gradients():
    torch.manual_seed(7)
    cfg = Qwen3Config(vocab_size=17, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, attention_dropout=0.)
    student = Qwen3ForCausalLM(cfg)
    other = deepcopy(student)
    teacher = Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    ids, mask, labels = collate([{"prompt_ids": [1, 2]}, {"prompt_ids": [4]}], [[3], [6, 7, 8, 9]], 0, "cpu")
    loss, n = response_loss(student, teacher, ids, mask, labels, chunk_size=2, equal_answers=True)
    sl = other(input_ids=ids, attention_mask=mask).logits[:, :-1]
    tl = teacher(input_ids=ids, attention_mask=mask).logits[:, :-1]
    expected = sl.new_zeros(())
    for i in range(2):
        valid = labels[i, 1:] != -100
        expected = expected + forward_kl(sl[i,valid], tl[i,valid]) / valid.sum()
    assert n == 5
    torch.testing.assert_close(loss, expected)
    (loss / 2).backward()
    (expected / 2).backward()
    for a, b in zip(student.parameters(), other.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)


def test_rotations_preserve_gold_and_cover_every_position():
    row = {"choices":{"text":["orange", "blue", "green"], "label":["1","2","3"]}, "answerKey":"2"}
    rows, orders = rotations([row])
    assert orders == [[0,1,2],[1,2,0],[2,0,1]]
    for variant, order in zip(rows, orders):
        pos = variant["choices"]["label"].index(variant["answerKey"])
        assert order[pos] == 1 and variant["choices"]["text"][pos] == "blue"
