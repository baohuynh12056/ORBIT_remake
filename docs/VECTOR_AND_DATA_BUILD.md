# Dataset and steering-vector construction

## Question datasets

ToneBank is loaded from the Hugging Face dataset documented in
`data/README.md`. TraitFactory's question-generation prompt, category list,
structured-output schema, cleaning, and JSONL writer are in
`data/traitfactory/generate_dataset.py`. The positive/negative trait prompts are
in `data/traitfactory/trait_definitions.py`.

## ORBIT vector files

`scripts/run_main_experiment.py` enables these pipeline options:

```text
--angular_auto_build_from_train
--angular_build_token_mode prompt_last
--angular_build_delta_proj_mode plane_mean
--caa_build_max_new_tokens 2048
--caa_build_generation_max_cap 2048
--angular_build_max_new_tokens 2048
--angular_build_generation_max_cap 2048
```

When `<label>_angular_instruct.pt` is missing, the pipeline calls
`_build_missing_angular_vectors_from_train` in
`scripts/run_experiment_pipeline.py`. For each attribute it:

1. combines an attribute-agnostic question with the positive and negative
   instruction prompts;
2. generates the paired responses with the target model;
3. uses `ResidualActivationExtractor` from `angular_residual_utils.py` to
   collect last-prompt-token activations at every LayerNorm extraction site;
4. computes the positive-minus-negative mean direction, orthogonal residual
   PCA direction, projected boost direction, calibration statistics, and
   extraction metadata for each module;
5. saves the `vectors` and `metadata` dictionaries to
   `<label>_angular_instruct.pt`.

At evaluation time, `angular_plane_mode=global` calls
`load_global_layernorm_vectors` in `angular_steering_utils.py`. It treats
the stored per-module first directions as cross-layer candidates, selects the
global first axis by maximum mean cosine similarity, obtains the second axis
from PC0 of the centered candidate directions and orthogonalizes it, then
broadcasts that global attribute plane to every LayerNorm intervention site.

The main launcher creates vectors below the user-provided output root:

```text
OUTPUT_ROOT/vectors/DATASET/MODEL_SLUG/prompt_last/global_plane_mean/
```
