# Dataset sources

## ToneBank

ToneBank is loaded from the following source:

- Dataset: https://huggingface.co/datasets/withmartian/tone_agnostic_questions
- Paper: https://aclanthology.org/2025.findings-emnlp.1278/

The experiment loader downloads that Hugging Face dataset when `--task tones`
is used without `--questions-file`. A local JSONL override may be provided;
each row must contain `text` and may contain `id` and `category`.

## TraitFactory

- `traitfactory/generate_dataset.py` generates trait-agnostic questions across
  the 20 categories used by the 1,077-question dataset.
- `traitfactory/trait_definitions.py` contains 13 traits, each with five
  positive and five negative instruction variants.

Generate a fresh question file with:

```bash
python data/traitfactory/generate_dataset.py \
  --output_dir outputs/traitfactory_data \
  --num_prompts 1200 \
  --openai_model gpt-4o-mini \
  --seed 42
```

Set `OPENAI_API_KEY` before running. The script keeps output local unless
`--upload_to_hub` is explicitly supplied.
