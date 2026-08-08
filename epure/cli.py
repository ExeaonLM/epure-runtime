"""Command line interface: `epure run`, `epure info`, `epure eval`.

Three verbs, all about running a model that is already compressed. There is no
`epure compress` and there will not be one — the encoder is not part of this
package.
"""
import argparse
import json
import sys
import time


def cmd_info(args):
    from .container import describe

    from .runtime import resolve
    info = describe(resolve(args.model))
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    width = max(len(k) for k in info)
    for k, v in info.items():
        print(f"  {k:<{width}}  {v}")
    return 0


def build_prompt(tok, text, raw=False):
    """Wrap a prompt in the model's chat template when it has one.

    Instruction-tuned models are trained to see their template and produce
    degenerate output without it -- Qwen3 answers "In one sentence, what is
    quantization?" with a run of digits when fed the bare string. Users type
    instructions, not completions, so the template is the right default and
    --raw is the escape hatch for base models and completion prompts.
    """
    if raw or not getattr(tok, "chat_template", None):
        return text
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True)


def cmd_run(args):
    from .runtime import load

    t0 = time.time()
    model, tok = load(args.model, device=args.device)
    print(f"  loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    prompt = args.prompt
    if prompt == "-":
        prompt = sys.stdin.read()

    ids = tok(build_prompt(tok, prompt, raw=args.raw),
              return_tensors="pt").input_ids.to(model.device)
    t0 = time.time()
    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=max(args.temperature, 1e-5),
        top_p=args.top_p,
        pad_token_id=tok.eos_token_id,
    )
    dt = time.time() - t0
    n_new = out.shape[1] - ids.shape[1]

    print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
    print(f"\n  {n_new} tokens in {dt:.1f}s  ({n_new/max(dt,1e-6):.1f} tok/s)",
          file=sys.stderr)
    return 0


def cmd_eval(args):
    """Benchmark via lm-eval, on the loaded model rather than a reloaded one.

    Reloading would evaluate the file instead of what is in memory, which
    silently hides any damage done by loading or by fine-tuning.
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("  needs the eval extra:  pip install 'epure-runtime[eval]'",
              file=sys.stderr)
        return 1

    from .runtime import load

    model, tok = load(args.model, device=args.device)
    for p in model.parameters():
        p.requires_grad_(False)

    res = lm_eval.simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tok,
                   batch_size=args.batch_size, device=model.device.type),
        tasks=args.tasks.split(","),
        limit=args.limit,
        batch_size=args.batch_size,
    )
    for task, r in sorted(res["results"].items()):
        acc = r.get("acc_norm,none", r.get("acc,none"))
        if acc is not None:
            print(f"  {task:<20} {100*acc:.2f}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="epure", description="Run compressed models.")
    p.add_argument("--version", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("model", help="path to a .ebin file, or a HF repo id")
    common.add_argument("--device", default=None, help="cpu / cuda (auto)")

    r = sub.add_parser("run", parents=[common], help="generate text")
    r.add_argument("--prompt", required=True, help="prompt, or - for stdin")
    r.add_argument("--max-new-tokens", type=int, default=256)
    r.add_argument("--temperature", type=float, default=0.6)
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--raw", action="store_true",
                   help="send the prompt verbatim, without the chat template")
    r.set_defaults(fn=cmd_run)

    i = sub.add_parser("info", parents=[common], help="describe a container")
    i.add_argument("--json", action="store_true")
    i.set_defaults(fn=cmd_info)

    e = sub.add_parser("eval", parents=[common], help="benchmark a model")
    e.add_argument("--tasks", default="arc_easy,hellaswag,piqa")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--batch-size", type=int, default=8)
    e.set_defaults(fn=cmd_eval)

    args = p.parse_args(argv)
    if args.version:
        from . import __version__
        print(__version__)
        return 0
    if not getattr(args, "fn", None):
        p.print_help()
        return 1
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
