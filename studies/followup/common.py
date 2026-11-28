import json
from pathlib import Path

RAW = Path("runs/followup")
OUT = Path("results/followup")
STUDY = Path("studies/followup")
ORIGINALS = ("Teacher", "E0", "E1", "E2", "E3", "E4")
RATES = {"lr-high": 1e-5, "lr-low": 3e-6}
POINTS = (32, 80, 157)
LR_MODELS = tuple(f"{arm}-{step}" for arm in RATES for step in POINTS)
REPLAY = ("replay-token-32", "replay-answer-32")


def load(path):
    return json.loads(Path(path).read_text())


def model_path(name, cfg):
    if name == "Teacher":
        return cfg["teacher"]
    if name == "E0":
        return cfg["student"]
    if name in ORIGINALS:
        return Path("runs") / name / "checkpoint"
    if name not in (*LR_MODELS, *REPLAY):
        raise ValueError(name)
    return RAW / "checkpoints" / name


def partition_rows(split):
    spec = load(STUDY / "split.json")
    rows = [json.loads(x) for x in Path("data/ifeval.jsonl").open()]
    ids = set(spec[split])
    selected = [r for r in rows if r["id"] in ids]
    assert [r["id"] for r in selected] == spec[split]
    return selected


def eval_rows(name, split):
    records = []
    for p in sorted((RAW / "ifeval" / split / name).glob("*.json")):
        records.extend(load(p))
    assert [r["id"] for r in records] == load(STUDY / "split.json")[split], (name, split)
    return records


def strict_instruction(rows):
    hits = sum(sum(r["inst_level_strict_acc"]) for r in rows)
    count = sum(len(r["inst_level_strict_acc"]) for r in rows)
    return hits / count
