"""Measure a compressed model and write a result file.

    pip install 'epure-runtime[eval]'
    python benchmarks/run.py <model> [--baseline <base>] [--tasks ...] [--limit N]

Produces `results/<model>.md`: footprint, decode throughput, peak memory,
accuracy and perplexity, plus the environment they were measured in. Accuracy
numbers are not comparable across harness versions or `--limit` values, so those
are recorded rather than assumed.
"""
import argparse
import json
import platform
import sys
import time
from datetime import date, timezone, datetime
from pathlib import Path

import torch

RESULTS = Path(__file__).parent / "results"
TASKS = "arc_easy,arc_challenge,hellaswag,piqa"


def environment():
    env = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
    else:
        env["cpu"] = platform.processor() or "unknown"
    try:
        import lm_eval
        env["lm_eval"] = lm_eval.__version__
    except Exception:
        pass
    from epure.kernels import cpu, cuda
    env["kernel"] = ("rust" if cpu.available() else "") + \
                    ("+triton" if cuda.available() else "") or "pytorch-fallback"
    return env


def throughput(model, tok, batches=(1, 8), new_tokens=64):
    """Tokens per second, after a warm-up.

    The first call pays for Triton autotuning and lazy CUDA init. Timing it
    would understate steady-state throughput badly on small models.
    """
    dev = model.device
    ids = tok("The memory wall is", return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        model.generate(ids, max_new_tokens=8, do_sample=False,
                       pad_token_id=tok.eos_token_id)

    out = {}
    for b in batches:
        x = ids.repeat(b, 1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        out[b] = (y.shape[1] - x.shape[1]) * b / dt
    return out


def accuracy(model, tok, tasks, limit, batch_size=8):
    """lm-eval on the model already in memory.

    Reloading would measure the file instead, hiding any damage done at load
    time or by a preceding fine-tune - the exact failure worth catching.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    for p in model.parameters():
        p.requires_grad_(False)
    res = lm_eval.simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size,
                   device=model.device.type),
        tasks=tasks.split(","), limit=limit, batch_size=batch_size)
    return {t: r.get("acc_norm,none", r.get("acc,none"))
            for t, r in sorted(res["results"].items())}


def measure(name, loader, args):
    print(f"\n  measuring {name}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, tok = loader()
    out = {"name": name, "load_seconds": round(time.time() - t0, 1)}
    out["throughput"] = throughput(model, tok)
    out["peak_gb"] = (torch.cuda.max_memory_allocated() / 2**30
                      if torch.cuda.is_available() else None)
    if not args.no_accuracy:
        out["accuracy"] = accuracy(model, tok, args.tasks, args.limit)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def render(rows, args, env):
    L = [f"# {rows[0]['name']}", "",
         "Produced by `benchmarks/run.py`. Committed unedited.", "",
         "## Environment", "", "| | |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in env.items()]

    L += ["", "## Decode throughput (tokens/second)", "",
          "| model | batch 1 | batch 8 |", "|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['throughput'][1]:.1f} | "
                 f"{r['throughput'][8]:.1f} |")

    if any(r.get("peak_gb") for r in rows):
        L += ["", "## Peak memory (GB)", "", "| model | peak |", "|---|---|"]
        L += [f"| {r['name']} | {r['peak_gb']:.2f} |" for r in rows
              if r.get("peak_gb")]

    if rows[0].get("accuracy"):
        tasks = list(rows[0]["accuracy"])
        L += ["", f"## Accuracy (lm-eval, limit={args.limit})", "",
              "| model | " + " | ".join(tasks) + " |",
              "|---" * (len(tasks) + 1) + "|"]
        for r in rows:
            vals = " | ".join(
                f"{100*r['accuracy'][t]:.2f}" if r["accuracy"].get(t) is not None
                else "-" for t in tasks)
            L.append(f"| {r['name']} | {vals} |")

    L += ["", "```json", json.dumps({"environment": env, "runs": rows}, indent=2),
          "```", ""]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", help=".ebin path or HF repo id")
    p.add_argument("--baseline", help="uncompressed model to compare against")
    p.add_argument("--tasks", default=TASKS)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--no-accuracy", action="store_true")
    args = p.parse_args()

    from epure import load

    env = environment()
    print("  environment:", ", ".join(f"{k}={v}" for k, v in env.items()))

    rows = [measure(args.model, lambda: load(args.model), args)]
    if args.baseline:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        def base():
            m = AutoModelForCausalLM.from_pretrained(
                args.baseline,
                torch_dtype=torch.float16 if dev == "cuda" else torch.float32
            ).to(dev).eval()
            return m, AutoTokenizer.from_pretrained(args.baseline)

        rows.append(measure(args.baseline, base, args))

    text = render(rows, args, env)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / (args.model.rstrip("/").split("/")[-1]
                      .replace(".ebin", "") + ".md")
    path.write_text(text, encoding="utf-8")
    print("\n" + text.split("```json")[0])
    print(f"  written to {path}")


if __name__ == "__main__":
    sys.exit(main())
