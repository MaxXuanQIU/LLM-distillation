import argparse

from kd.common import config, load_model, read_jsonl, seed_all, tokenizer, write_json
from kd.evaluate import evaluate_arc, score_ifeval_batch
from studies.followup.common import ORIGINALS, OUT, RAW, load, model_path, partition_rows


def rotations(rows):
    rotated, orders = [], []
    for row in rows:
        texts, labels = row["choices"]["text"], row["choices"]["label"]
        for shift in range(len(texts)):
            order = list(range(shift, len(texts))) + list(range(shift))
            rotated.append({**row, "choices": {"text": [texts[i] for i in order],
                                               "label": [labels[i] for i in order]}})
            orders.append(order)
    return rotated, orders


def arc(name, cfg):
    dest = OUT / "arc" / f"{name}.json"
    if dest.exists():
        return
    path = model_path(name, cfg)
    model, tok = load_model(path, "cuda:0"), tokenizer(path)
    source = read_jsonl("data/arc.jsonl")
    rotated, orders = rotations(source)
    answers = {r["id"]: r["choices"]["label"].index(r["answerKey"]) for r in source}
    by_id = {key: [] for key in answers}
    for row, order in zip(evaluate_arc(model, tok, rotated), orders, strict=True):
        pos = ord(row["prediction"]) - ord("A")
        content = order[pos]
        assert row["correct"] == (content == answers[row["id"]])
        by_id[row["id"]].append({"position": pos, "content": content, "correct": row["correct"],
                                 "letter_logprobs": row["choice_logprobs"]})
    assert len(rotated) == 1599 and len(by_id) == 400
    if name in ORIGINALS:
        old = read_jsonl(f"results/{name}/arc.jsonl")
        assert len(old) == len(by_id)
        for row in old:
            again = by_id[row["id"]][0]
            assert again["position"] == ord(row["prediction"]) - ord("A"), (name, row["id"])
    write_json(dest, {"model": name, "gold_content": answers, "rotations": by_id})
    print(name, len(rotated), "rotations", flush=True)


def ifeval(name, split, shard, shards, cfg):
    rows = partition_rows(split)
    size = cfg["eval_batch_size"]
    dest = RAW / "ifeval" / split / name
    pending = [(i, rows[j:j+size]) for i, j in enumerate(range(0, len(rows), size))
               if i % shards == shard and not (dest / f"{i:03d}.json").exists()]
    if not pending:
        return
    path = model_path(name, cfg)
    model, tok = load_model(path, "cuda:0"), tokenizer(path)
    for i, batch in pending:
        records = list(score_ifeval_batch(model, tok, batch, cfg))
        assert [r["id"] for r in records] == [r["id"] for r in batch]
        write_json(dest / f"{i:03d}.json", records)
        print(name, split, i, len(records), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=("arc", "ifeval"))
    p.add_argument("name")
    p.add_argument("--split", choices=("screen", "confirm"), default="screen")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    args = p.parse_args()
    cfg = config()
    seed_all(cfg["seed"])
    if args.task == "arc":
        arc(args.name, cfg)
    else:
        ifeval(args.name, args.split, args.shard, args.shards, cfg)
