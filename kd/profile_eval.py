"""Memory-only pilot: longest IFEval prompts, forced maximum generation length.

No benchmark scores are computed; these forced continuations are not evaluation
outputs and must never be reused as such.
"""
import time
import torch
from kd.common import config, load_model, pad_prompts, read_jsonl, seed_all, tokenizer, write_json


def main():
    cfg = config()
    seed_all(cfg["seed"])
    model = load_model(cfg["teacher"], "cuda:0")
    tok = tokenizer(cfg["teacher"])
    rows = sorted(read_jsonl("data/ifeval.jsonl"), key=lambda r: len(r["prompt_ids"]), reverse=True)[:cfg["eval_batch_size"]]
    batch = pad_prompts(rows, tok, "cuda:0")
    start = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**batch, min_new_tokens=cfg["eval_max_new_tokens"],
            max_new_tokens=cfg["eval_max_new_tokens"], do_sample=False, use_cache=True,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    result = {"batch_size": len(rows), "prompt_width": batch.input_ids.shape[1],
              "generated_width": output.shape[1]-batch.input_ids.shape[1],
              "seconds": time.monotonic()-start, "peak_memory_gb": torch.cuda.max_memory_allocated()/1e9,
              "purpose": "worst-length memory pilot only; no benchmark scores"}
    write_json("runs/pilot_eval_memory.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
