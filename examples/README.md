# Examples

```bash
pip install epure-runtime
```

| file | what it shows |
|---|---|
| [`01_generate.py`](01_generate.py) | text generation, and that resident memory tracks the *compressed* size |
| [`02_finetune.py`](02_finetune.py) | training in the compressed state, with a check that indices never moved |
| [`03_benchmark.py`](03_benchmark.py) | footprint, decode throughput and accuracy — the numbers for a model card |

```bash
python examples/01_generate.py
python examples/01_generate.py Exeaon/Exeaon-Dzo-0.6B "Write a haiku about latency."
python examples/02_finetune.py
pip install 'epure-runtime[eval]' && python examples/03_benchmark.py
```

Each accepts a model as the first argument — a Hugging Face repo id, or a local
`.ebin` path.
