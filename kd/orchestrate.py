"""Resumable job graph with an immutable, maximum four-GPU allocation."""
import argparse
import concurrent.futures
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from kd.common import config, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "generate", "pilot", "research"], required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--wait", action="store_true", help="Wait for another phase to release the four GPU locks")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    assert len(gpus) == 4 and len(set(gpus)) == 4
    assert all(g in {"0", "1", "2", "3"} for g in gpus), "Only the allocated GPUs 0-3 may be used"
    locks = []
    for gpu in gpus:
        lock = open(f"/tmp/llm-distillation-{os.getuid()}-gpu{gpu}.lock", "w")
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait else fcntl.LOCK_NB))
        locks.append(lock)
    Path("logs").mkdir(exist_ok=True)
    state = {"phase": args.phase, "started": time.time(), "gpus": gpus, "jobs": {}}
    import threading
    state_lock = threading.Lock()

    def job(name, command, devices):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=",".join(devices), TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1",
                   HF_ENDPOINT="https://hf-mirror.com", HF_HUB_DISABLE_XET="1",
                   HF_HOME=str(Path(".cache/huggingface").resolve()),
                   NLTK_DATA=str(Path.home()/"nltk_data"),
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        with state_lock:
            state["jobs"][name] = {"status": "running", "started": time.time(), "gpus": devices}
            write_json("logs/status.json", state)
        with open(f"logs/{name}.log", "a") as log:
            result = subprocess.run([sys.executable, "-m", *command], env=env, stdout=log, stderr=subprocess.STDOUT)
        with state_lock:
            state["jobs"][name].update(status="complete" if result.returncode == 0 else "failed",
                                       finished=time.time(), returncode=result.returncode)
            write_json("logs/status.json", state)
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect logs/{name}.log")

    def parallel(jobs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fn) for fn in jobs]
            for future in futures:
                future.result()

    def train(exp, devices):
        if not Path(f"runs/{exp}/summary.json").exists():
            job(f"train_{exp}", ["kd.train", "--experiment", exp, "--resume"], devices)

    def evaluate(name, device):
        if not Path(f"results/{name}/metrics.json").exists():
            job(f"eval_{name}", ["kd.evaluate", "--name", name], [device])

    if args.phase == "prepare":
        job("prepare", ["kd.prepare"], [])
    elif args.phase == "generate":
        parallel([lambda i=i: job(f"generate_{i}", ["kd.generate", "--shard", str(i)], [gpus[i]]) for i in range(4)])
        job("merge", ["kd.generate", "--merge"], [])
        job("audit_data", ["kd.audit", "--data-only"], [])
    elif args.phase == "pilot":
        job("audit_data", ["kd.audit", "--data-only"], [])
        parallel([lambda: job("pilot_E1", ["kd.train", "--experiment", "E1", "--pilot-steps", "4"], [gpus[2]]),
                  lambda: job("pilot_E4", ["kd.train", "--experiment", "E4", "--pilot-steps", "4"], gpus[:2])])
    else:
        def first_pair():
            def first_student():
                train("E1", [gpus[0]])
                evaluate("E0", gpus[0])
                evaluate("E1", gpus[0])
            parallel([first_student, lambda: evaluate("Teacher", gpus[1])])
            train("E3", gpus[:2])
            evaluate("E3", gpus[0])
        def second_pair():
            train("E4", gpus[2:])
            train("E2", gpus[2:])
            parallel([lambda: evaluate("E4", gpus[2]), lambda: evaluate("E2", gpus[3])])
        parallel([first_pair, second_pair])
        job("audit", ["kd.audit"], [])
        job("report", ["kd.report"], [])
    state["finished"] = time.time()
    write_json("logs/status.json", state)


if __name__ == "__main__":
    main()
