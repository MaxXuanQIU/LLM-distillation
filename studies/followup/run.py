"""Run the three studies with exclusive assignments on physical GPUs 0 through 3."""
import concurrent.futures
import fcntl
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
import threading
import time

from kd.common import config, digest_file, write_json
from studies.followup.common import (LR_MODELS, ORIGINALS, OUT, RATES, RAW, REPLAY, STUDY,
                                     eval_rows, load, strict_instruction)

logdir = Path("logs/followup")
mutex = threading.Lock()
jobs = {}


def run(module, arguments, devices):
    key = "_".join([module, *map(str, arguments)])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices)), "OMP_NUM_THREADS": "4",
           "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "NLTK_DATA": str(Path.home()/"nltk_data")}
    job = {"started": time.time(), "devices": devices, "status": "running"}
    with mutex:
        jobs.setdefault(key, []).append(job)
        write_json(logdir/"schedule.json", jobs)
    with (logdir / f"{key}.log").open("a") as log:
        proc = subprocess.run([sys.executable, "-m", "studies.followup."+module, *map(str,arguments)],
                              stdout=log, stderr=subprocess.STDOUT, env=env)
    with mutex:
        job.update(finished=time.time(), status="failed" if proc.returncode else "complete", code=proc.returncode)
        write_json(logdir/"schedule.json", jobs)
    if proc.returncode:
        raise RuntimeError(key)


def parallel(tasks):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fn) for fn in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def singles(commands):
    devices = Queue()
    for gpu in range(4):
        devices.put(gpu)
    def execute(command):
        gpu = devices.get()
        try:
            run("measure", command, [gpu])
        finally:
            devices.put(gpu)
    parallel([lambda c=c: execute(c) for c in commands])


def audit_inputs():
    audit = load("results/audit.json")
    assert config()["eval_max_new_tokens"] == config()["max_completion_tokens"] == 16384
    expected = dict(audit["data_sha256"])
    expected["runs/E4/trajectories.jsonl"] = load("results/artifact_hashes.json")["runs/E4/trajectories.jsonl"]
    for n, entry in audit["experiments"].items():
        for filename, sha in entry["checkpoint_sha256"].items():
            expected[f"runs/{n}/checkpoint/{filename}"] = sha
    for kind, files in load("data/manifest.json")["model_files_sha256"].items():
        for f, sha in files.items():
            expected[str(Path(config()[kind])/f)] = sha
    for path, sha in expected.items():
        assert digest_file(path) == sha, path
    split = load(STUDY/"split.json")
    assert split["ifeval_sha256"] == expected["data/ifeval.jsonl"]
    assert not set(split["screen"]) & set(split["confirm"])
    write_json(OUT/"inputs.json", {"sha256": expected, "split_sha256": digest_file(STUDY/"split.json"),
                                   "config_sha256": digest_file("config.json"),
                                   "revision": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()})


def select():
    ranking = []
    for name in LR_MODELS:
        rows = eval_rows(name, "screen")
        arm, step = name.rsplit("-", 1)
        ranking.append((name, strict_instruction(rows), sum(r["truncated"] for r in rows), int(step), RATES[arm]))
    ranking.sort(key=lambda r:(-r[1], r[2], r[3], r[4]))
    reference = "lr-high-157"
    challenger = next(r[0] for r in ranking if r[0] != reference)
    write_json(OUT/"selection.json", {"ranking": ranking, "winner": ranking[0][0],
                                     "challenger": challenger, "reference": reference})
    return ["E0", reference, challenger, *REPLAY]


def main():
    logdir.mkdir(parents=True, exist_ok=True)
    locks = []
    for gpu in range(4):
        lock = open(f"/tmp/llm-distillation-{os.getuid()}-gpu{gpu}.lock", "w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    if (logdir/"schedule.json").exists():
        jobs.update(load(logdir/"schedule.json"))
    audit_inputs()
    singles([["arc", name] for name in ORIGINALS])
    def pair(first, second, devices):
        for arm in (first, second):
            run("train", [arm], devices)
    parallel([lambda: pair("lr-high", "replay-token", [0,1]),
              lambda: pair("lr-low", "replay-answer", [2,3])])
    singles([["arc", name] for name in (*LR_MODELS, *REPLAY)])
    screen = ["E0", *LR_MODELS, *REPLAY]
    singles([["ifeval", n, "--split", "screen", "--shard", s, "--shards", 2]
             for s in range(2) for n in screen])
    confirm = select()
    singles([["ifeval", n, "--split", "confirm", "--shard", s, "--shards", 4]
             for s in range(4) for n in confirm])
    for name in confirm:
        eval_rows(name, "confirm")
    write_json(OUT/"completion.json", {"finished": time.time(), "screen": screen, "confirm": confirm})


if __name__ == "__main__":
    main()
