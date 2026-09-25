# cuad-qlora-vllm

Fine-tune **Qwen3-4B-Instruct** with **QLoRA (Unsloth)** to extract legal clauses from contracts in the
[CUAD](https://github.com/The-Atticus-Project/cuad) dataset, and measure how much fine-tuning gains over
zero-shot and retrieval (RAG) baselines. Evals run on **vLLM**.

Results are reported with the same metrics as
[ContractEval](https://arxiv.org/abs/2508.03080), so they compare directly with published numbers
(GPT-4.1: 0.641 F1, Claude Sonnet 4: 0.523 F1 on the CUAD test set).

> **Status:** Phase 1. The data pipeline, metrics, inference runner and hybrid retrieval are done;
> baseline runs need a GPU (vLLM). See [docs/plan.md](docs/plan.md) for the roadmap.

## Task

For each contract and each of CUAD's 41 clause categories (e.g. *Governing Law*, *Non-Compete*,
*Cap On Liability*), the model returns every relevant passage verbatim, or `[]` if the clause is absent.
About 70% of (contract, category) pairs are absent, so saying "not present" correctly matters as much
as extracting.

## Design decisions

- **Eval set = the official CUAD test split** (102 contracts, 4,182 pairs), the same set ContractEval
  uses. 40 contracts from the official train split are held out for validation, leaving 368 for training.
- **Chunked training and inference.** Contracts run up to ~300k characters, too long to fine-tune a 4B
  model on within budget. Contracts are split into ~3k-token overlapping windows; per-window answers
  are merged back into contract-level spans before scoring, so metrics stay comparable to
  full-context models.
- **Negative sampling.** Almost all (chunk, category) pairs are empty. Training keeps every positive
  and samples negatives at a configurable ratio (`negatives_per_positive` in `configs/data.yaml`).

## Metrics

All metrics are computed over (contract, category) pairs; see [metrics.py](src/cuad_llm/metrics.py).

| Metric | Meaning |
| --- | --- |
| `coverage_f1` / `coverage_f2` | ContractEval's headline: TP only if the prediction fully covers every gold span |
| `jaccard` | ContractEval's token-set overlap on gold-positive pairs |
| `detection_f1` | Did the model correctly say the clause is present or absent? |
| `token_f1` | SQuAD-style token overlap on gold-positive pairs |
| `laziness` | Share of present clauses the model called absent |
| `verbatim` | Share of predicted quotes that really appear in the contract (hallucination check) |

F1 scores come with 95% bootstrap confidence intervals, resampled over contracts.

## Quickstart

Local steps (data prep, tests) run on any machine. Training and model evals need a CUDA GPU.

```bash
uv sync                                # Python 3.12 environment
uv run pytest                          # unit tests (add --group rag to include retrieval tests)
uv run python scripts/prepare_data.py  # download CUAD, split, build SFT data into data/processed/
```

Baselines (any OpenAI-compatible endpoint; `--dry-run` sizes the job without calling a model,
`--limit 3` runs a smoke test on three contracts):

```bash
# On the GPU box, start vLLM for the model under test, e.g.:
#   vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 16384 --enable-prefix-caching
# Zero-shot Qwen3-4B-Instruct: the model before fine-tuning
uv run python scripts/predict.py --config configs/zeroshot_qwen3_4b.yaml
# Baseline A: zero-shot Qwen3-8B (serve Qwen/Qwen3-8B instead)
uv run python scripts/predict.py --config configs/zeroshot_qwen3_8b.yaml
# Baseline B: hybrid BM25 + dense retrieval, one request per (contract, category)
uv run --group rag python scripts/predict.py --config configs/rag_qwen3_8b.yaml
# Score runs side by side with the published ContractEval numbers
uv run python scripts/evaluate.py outputs/zeroshot_qwen3_4b outputs/zeroshot_qwen3_8b outputs/rag_qwen3_8b
```

Responses are cached in `outputs/<run>/responses.jsonl`, so an interrupted run resumes where it stopped.

## Running on Kaggle (free GPUs)

Training and GPU evals run on Kaggle's free 2x T4 notebooks. The repo must be on GitHub (public) so
the notebooks can clone it.

1. **Train:** upload [notebooks/kaggle_train.ipynb](notebooks/kaggle_train.ipynb) to Kaggle, set
   `REPO_URL`, run a 20-step smoke test, then commit a full run. Instructions are in the notebook.
2. **Evaluate:** upload [notebooks/kaggle_eval.ipynb](notebooks/kaggle_eval.ipynb), attach the training
   output, and run the tasks: baselines, checkpoint selection on val, then the final test eval.

## Layout

```
configs/          YAML configs per stage
scripts/          entry points: prepare_data.py, train.py, predict.py, evaluate.py,
                  start_vllm.sh
notebooks/        Kaggle notebooks for training and evaluation
src/cuad_llm/     shared library: data, chunking, prompts, spans, metrics, inference,
                  retrieval, pipeline
tests/            unit tests
docs/plan.md      roadmap and stack rationale
```

## Data licence

CUAD is released by The Atticus Project under CC BY 4.0.
