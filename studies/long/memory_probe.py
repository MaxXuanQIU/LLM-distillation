"""Exercise full-length forward KL and AdamW, including existing optimizer state."""
import torch
from kd.common import collate, config, load_model, read_jsonl, seed_all, tokenizer, write_json
from kd.loss import response_loss

cfg = config()
seed_all(cfg["seed"])
student = load_model(cfg["student"], "cuda:0", train=True)
teacher = load_model(cfg["teacher"], "cuda:1")
tok = tokenizer(cfg["student"])
row = max(read_jsonl("data/validation_prompts.jsonl"), key=lambda r: len(r["prompt_ids"]))
pattern = tok.encode(" A memory check uses a synthetic token sequence.", add_special_tokens=False)
optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["learning_rate"], foreach=False)
for size in [8, cfg["max_completion_tokens"]]:
    completion = (pattern * (size // len(pattern) + 1))[:size]
    ids, mask, labels = collate([row], [completion], tok.pad_token_id, student.device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, count = response_loss(student, teacher, ids, mask, labels, cfg["loss_chunk_size"])
    assert count == size and torch.isfinite(loss)
    (loss/count).backward()
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["max_grad_norm"])
    assert torch.isfinite(norm)
    optimizer.step()
write_json("long_pilot/memory.json", {"response_tokens":count, "prompt_tokens":len(row["prompt_ids"]),
           "peak_gb":[torch.cuda.max_memory_allocated(i)/1e9 for i in range(2)]})
