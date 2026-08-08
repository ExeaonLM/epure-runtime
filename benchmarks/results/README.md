# Results

Measured output for every published Exeaon model. Each file is written by
[`../run.py`](../run.py) and committed unedited, including results that are
unfavourable.

Model cards link here rather than restating numbers, so there is one source of
truth and no opportunity for a card to drift from what was measured.

## Published models

| model | base | size | ratio | accuracy delta | results |
|---|---|---|---|---|---|
| _none yet_ | | | | | |

<!--
  Add a row when a model is published. "accuracy delta" is the mean change
  across the four lm-eval tasks against the base model measured on the same
  hardware with the same harness version - not against a number from someone
  else's README.
-->

## Reproducing

```bash
pip install 'epure-runtime[eval]'
python benchmarks/run.py <model> --baseline <base-model>
```

The environment block at the top of each result file records hardware, driver,
torch and `lm-eval` versions. Accuracy figures are not comparable across
different `lm-eval` versions or `limit` values, so those are recorded rather
than assumed.

## Standing caveats

- `--limit` subsets the evaluation set. Anything below the full set carries
  sampling error of roughly one to three points; the limit used is in every file.
- Decode throughput depends on batch size, context length and hardware. A single
  tokens-per-second figure without those three is not meaningful, so all three
  are recorded.
- Perplexity and multiple-choice accuracy disagree regularly. A model can hold
  perplexity while losing several points of ARC-Challenge. Both are published.
