import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def config(path="config.json"):
    return json.loads(Path(path).read_text())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tok.padding_side = "left"
    return tok


def prompt_ids(tok, text):
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )


def load_model(path, device, train=False):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32 if train else torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    ).to(device)
    if train:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    else:
        model.requires_grad_(False)
        model.eval()
    return model


def pad_prompts(rows, tok, device):
    return tok.pad({"input_ids": [r["prompt_ids"] for r in rows]},
                   padding=True, return_tensors="pt").to(device)


def trim_completion(ids, eos):
    ids = list(ids)
    if eos in ids:
        return ids[:ids.index(eos) + 1]
    return ids


@torch.no_grad()
def _generate_batch(model, tok, rows, max_new_tokens, cfg, sample=True):
    training = model.training
    model.eval()
    batch = pad_prompts(rows, tok, model.device)
    kw = {"do_sample": sample}
    if sample:
        kw.update(temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(**batch, max_new_tokens=max_new_tokens,
                                use_cache=True, pad_token_id=tok.pad_token_id,
                                eos_token_id=tok.eos_token_id, **kw)
    completions = [trim_completion(x, tok.eos_token_id) for x in
                   output[:, batch.input_ids.shape[1]:].tolist()]
    model.train(training)
    return completions


def generate(model, tok, rows, max_new_tokens, cfg, sample=True):
    """Continue unfinished sequences in smaller batches as their context grows."""
    limit = cfg.get("max_generation_batch_tokens")
    if limit is None:
        return _generate_batch(model, tok, rows, max_new_tokens, cfg, sample)
    completions = [[] for _ in rows]
    active = list(range(len(rows)))
    generated = 0
    while active and generated < max_new_tokens:
        stop = min(max_new_tokens, max(512, generated * 2))
        longest = max(len(rows[i]["prompt_ids"]) for i in active)
        capacity = max(1, limit // (longest + stop))
        remaining = []
        for offset in range(0, len(active), capacity):
            indices = active[offset:offset + capacity]
            inputs = [{"prompt_ids": rows[i]["prompt_ids"] + completions[i]} for i in indices]
            tails = _generate_batch(model, tok, inputs, stop - generated, cfg, sample)
            for i, tail in zip(indices, tails):
                completions[i].extend(tail)
                if tail[-1] != tok.eos_token_id:
                    assert len(completions[i]) == stop
                    remaining.append(i)
        generated = stop
        active = remaining
    return completions


def training_parts(rows, completions, token_budget):
    """Pack nearby lengths within an update. A long single sequence stays intact."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt_ids"]) + len(completions[i]))
    batch = []
    for i in order:
        length = len(rows[i]["prompt_ids"]) + len(completions[i])
        if batch and length * (len(batch) + 1) > token_budget:
            yield [rows[j] for j in batch], [completions[j] for j in batch]
            batch = []
        batch.append(i)
    if batch:
        yield [rows[j] for j in batch], [completions[j] for j in batch]


def collate(rows, completions, pad_id, device):
    """Right padding for training; mask predictions, including the genuine EOS."""
    seqs = [r["prompt_ids"] + c for r, c in zip(rows, completions)]
    n = max(map(len, seqs))
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, (row, comp, seq) in enumerate(zip(rows, completions, seqs)):
        p = len(row["prompt_ids"])
        ids[i, :len(seq)] = torch.tensor(seq, device=device)
        attention[i, :len(seq)] = 1
        labels[i, p:len(seq)] = torch.tensor(comp, device=device)
    return ids, attention, labels


def provenance():
    import importlib.metadata
    versions = {}
    for pkg in ["torch", "transformers", "trl", "datasets", "lm_eval", "accelerate"]:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "versions": versions, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
