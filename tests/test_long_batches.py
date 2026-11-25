from types import SimpleNamespace

from kd import common


def test_continuations_keep_order_and_stop_on_eos(monkeypatch):
    target_lengths = {1: 20, 2: 512, 3: 558, 4: 1100}

    def generate_piece(model, tok, rows, budget, cfg, sample):
        pieces = []
        for row in rows:
            first, *prefix = row["prompt_ids"]
            assert prefix == [first] * len(prefix)
            remaining = target_lengths[first] - len(prefix)
            assert remaining > 0
            tokens = [first] * min(budget, remaining)
            if remaining <= budget:
                tokens[-1] = tok.eos_token_id
            pieces.append(tokens)
        return pieces

    monkeypatch.setattr(common, "_generate_batch", generate_piece)
    rows = [{"prompt_ids": [i]} for i in target_lengths]
    answers = common.generate(None, SimpleNamespace(eos_token_id=0), rows, 1040,
                              {"max_generation_batch_tokens": 1500})
    assert [len(a) for a in answers] == [20, 512, 558, 1040]
    assert [a[-1] for a in answers] == [0, 0, 0, 4]


def test_training_token_budget_keeps_single_long_sequence():
    rows = [{"prompt_ids": [1], "id": i} for i in range(4)]
    completions = [[2] * n for n in (90, 3, 6, 4)]
    pieces = list(common.training_parts(rows, completions, 16))
    assert sorted((r["id"], len(c)) for batch, targets in pieces
                  for r, c in zip(batch, targets)) == list(enumerate([90, 3, 6, 4]))
    assert all(len(batch) == 1 or len(batch)*max(1+len(c) for c in targets) <= 16 for batch, targets in pieces)
