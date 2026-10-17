# ORBIT code

This package contains the implementation for the ORBIT ToneBank,
TraitFactory, and TinyBench experiments.

## Layout

- `angular_steering_utils.py` and `angular_residual_utils.py`: vector loading,
  global-plane construction from per-module candidates, activation extraction,
  and steering hooks.
- `scripts/run_main_experiment.py`: model-agnostic launcher for either
  benchmark.
- `scripts/run_experiment_pipeline.py`: end-to-end vector building,
  calibration, generation, judging, and aggregation pipeline.
- `scripts/run_tinybench.sh`: TinyBench launcher that evaluates all
  unordered attribute combinations for every requested `K`.
- `ksteering/`: K-Steering code from the
  [authors' repository](https://github.com/withmartian/k-steering), used under
  its original license. The pipeline trains the K-Steering classifier on the
  shared contrastive data; the surrounding orchestration and dataset integration
  are adapted for these experiments.
- `tasks/tinyBenchmarks_gen/`: TinyBench task definitions used by `lm-eval`.
- `scripts/judge_responses.py`: standalone definition-based response judge.
- `data/traitfactory/`: TraitFactory generation script and the 13-trait
  positive/negative prompt definitions.
- `docs/VECTOR_AND_DATA_BUILD.md`: dataset and steering-vector construction
  details.

## Environment

Python 3.10 was used. Create an environment and install dependencies with:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Install a PyTorch wheel appropriate for the local CUDA driver if the pinned
wheel is unsuitable. Hugging Face access is required for gated models.
Set `OPENAI_API_KEY` when running calibration and judging. GSM8K evaluation
uses `math_verify`, which is included in `requirements.txt`.

## Datasets

By default, the code loads ToneBank from
[`withmartian/tone_agnostic_questions`](https://huggingface.co/datasets/withmartian/tone_agnostic_questions),
the 1,184-question source associated with the
[K-Steering paper](https://aclanthology.org/2025.findings-emnlp.1278/).
An explicit local JSONL can instead be supplied with `--questions-file`.

TraitFactory is generated locally:

```bash
.venv/bin/python data/traitfactory/generate_dataset.py \
  --output_dir outputs/traitfactory_data \
  --num_prompts 1200 \
  --openai_model gpt-4o-mini \
  --seed 42
```

The generator writes `outputs/traitfactory_data/dataset.jsonl`. Pass this file
through `--questions-file` and pass
`data/traitfactory/trait_definitions.py` through
`--trait-definitions-file`. Each question row contains `id`, `text`, and
`category`. Dataset generation is stochastic even with a fixed seed because it
calls a hosted model. The TraitFactory dataset used in the experiments contains
1,077 accepted rows.

Saved responses can be judged separately from generation:

```bash
.venv/bin/python scripts/judge_responses.py \
  --input-csv outputs/main/responses_scored.csv \
  --output-dir outputs/main/judged_responses \
  --task tones \
  --judge-type target
```

`target` is the paper's JSON-only attribute judge and is the default. Use
`--judge-type detailed` for the longer rubric with written reasoning, or
`--judge-type both` to run both judges.

## Main experiments

The launcher runs all `binom(5,K)` attribute combinations for `K=1,2,3`.
For example:

```bash
.venv/bin/python scripts/run_main_experiment.py \
  --model-name MODEL_ID \
  --task tones \
  --output-root outputs/main \
  --device-ids 0
```

For TraitFactory, add:

```bash
--task traits \
--questions-file outputs/traitfactory_data/dataset.jsonl \
--trait-definitions-file data/traitfactory/trait_definitions.py
```

Use `--dry-run` to validate and print the resolved command without loading a
model.

The public experiment defaults are:

- one global cross-layer plane per attribute, constructed from per-module
  candidate directions and applied at every LayerNorm intervention site;
- full alignment, equivalent to a 0-degree final target angle;
- adaptive zero-margin gate `g_k=max(0,-s_k)` for rotation and boost;
- response-token intervention;
- up to 2,048 tokens for contrastive-response generation;
- temperature sampling with temperature 0.7 and top-p 0.9;
- every unordered attribute combination for each requested `K`.

Steering-vector files are built automatically when absent. The launcher enables
`--angular_auto_build_from_train`; the pipeline generates paired positive and
negative responses, extracts last-prompt-token activations at every LayerNorm
site, computes the per-module candidate statistics, and saves one
`<label>_angular_instruct.pt` file per attribute. See
`docs/VECTOR_AND_DATA_BUILD.md` for the exact code path and file contents.

## TinyBench

TinyBench consumes a vector directory and calibration CSV from a completed
main run. Every unordered label combination is evaluated for each requested
`K`.

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_tinybench.sh \
  --model-name MODEL_ID \
  --vector-dir PATH_TO_VECTORS \
  --calibration-csv PATH_TO_CALIBRATION_CSV \
  --labels LABEL_1,LABEL_2,LABEL_3,LABEL_4,LABEL_5 \
  --output-dir outputs/tinybench \
  --gpu 0
```

The six tasks are MMLU, HellaSwag, ARC, GSM8K, TruthfulQA, and WinoGrande.
GSM8K is scored with `math_verify`. Use `--dry-run` to list all combinations
without launching evaluation.
