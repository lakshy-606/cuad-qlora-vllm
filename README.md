# cuad-qlora-vllm

Fine-tuning **Qwen3-4B-Instruct** with **QLoRA (Unsloth)** to extract legal clauses from contracts in
the [CUAD](https://github.com/The-Atticus-Project/cuad) dataset, evaluated with **vLLM** against
zero-shot, larger-model and retrieval (RAG) baselines. Everything ran on Kaggle's free T4 GPUs.

## Results

Official CUAD test split: 102 contracts × 41 clause categories = 4,182 pairs. Metrics follow
[ContractEval](https://arxiv.org/abs/2508.03080); 95% confidence intervals come from a bootstrap over
contracts.

| Model | Coverage F1 | Precision | Recall | Detection F1 | Missed clauses | Verbatim quotes | Token F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Qwen3-4B, fine-tuned (this repo)** | **0.714** [0.688, 0.738] | 0.718 | **0.711** | **0.856** | **4.3%** | 98.8% | **0.751** |
| Qwen3-4B, zero-shot | 0.642 [0.616, 0.667] | 0.727 | 0.575 | 0.803 | 18.4% | 87.6% | 0.499 |
| Qwen3-4B + hybrid RAG | 0.592 [0.557, 0.628] | 0.740 | 0.493 | 0.764 | 27.5% | 86.3% | 0.478 |
| Qwen3-8B, zero-shot | 0.512 [0.482, 0.544] | 0.783 | 0.380 | 0.713 | 38.8% | 94.4% | 0.410 |

Paired bootstrap over contracts (10,000 resamples), fine-tuned minus baseline:

| Baseline | Coverage F1 difference [95% CI] | Detection F1 difference [95% CI] |
| --- | --- | --- |
| Qwen3-4B zero-shot | **+0.073** [+0.049, +0.095] | +0.053 [+0.038, +0.068] |
| Qwen3-4B + RAG | **+0.123** [+0.095, +0.151] | +0.092 [+0.073, +0.112] |
| Qwen3-8B zero-shot | **+0.202** [+0.168, +0.235] | +0.144 [+0.119, +0.168] |

No resample showed the fine-tuned model failing to improve on any baseline (p < 0.0001).

**What fine-tuning changed**

- **Recall rose from 0.575 to 0.711 at unchanged precision** (0.727 → 0.718): the model finds far more
  of each clause without extracting more on contracts where the clause is absent.
- **Missed clauses fell from 18.4% to 4.3%**, and **quotes copied verbatim rose from 87.6% to 98.8%**:
  the untuned model gives up or paraphrases; the tuned one extracts and copies exactly.
- **Better in 29 of 41 clause categories**, worse in 4. The largest gains are on hard, rarer clauses,
  e.g. Minimum Commitment 0.11 → 0.52, Notice Period to Terminate Renewal 0.44 → 0.86.
- **Retrieval didn't help a 4B model:** asking once over the top 16 retrieved passages, instead of once
  per window, lowered recall (0.575 → 0.493). Retrieval surfaces only 83% of gold spans, which caps
  recall before the model reads anything.
- **Twice the parameters didn't help either:** zero-shot Qwen3-8B (thinking disabled) marked 39% of
  present clauses as absent.

**How these compare with published numbers.** ContractEval reports coverage F1 0.641 for GPT-4.1 and
0.523 for Claude Sonnet 4 on the same test split. The metric and split match, but the setup does not:
models here read ~3,000-token windows whose answers are merged per contract, and that alone lifts
recall. Zero-shot Qwen3-4B scoring level with GPT-4.1 (0.642 vs 0.641) is a sign of this, so treat the
published figures as context, not a head-to-head result.

## A data bug the error analysis caught

The first fine-tune scored **0.585** coverage F1, *below* the untuned model, despite fewer missed
clauses and better span overlap. Breaking the failures down ([analyze_errors.py](scripts/analyze_errors.py))
showed 80% were clauses where the model returned one of several gold spans, and the *Parties*
category had collapsed (1 of 102 contracts fully covered).

The cause: CUAD's training file stores each answer as a separate question, and the loader overwrote
repeated categories, so **every training target kept only its last span**, while 39% of test pairs have
several. The test file was unaffected. After the fix (verified against `CUADv1.json` for all 16,728
training pairs, with a regression test) and retraining, coverage F1 rose to **0.714** (+0.129). Detection
F1 barely moved (+0.007, not significant), consistent with the bug affecting span completeness rather
than clause spotting.

## Method

**Task.** For each contract and clause category (e.g. *Governing Law*, *Non-Compete*, *Cap On
Liability*), return every relevant passage verbatim as a JSON list, or `[]` if the clause is absent.
About 70% of pairs are absent.

**Data.** The official CUAD test split is the eval set; 40 contracts from the train split are held out
for validation (seed 42), leaving 368 for training. Contracts (up to ~300k characters) are cut into
12,000-character windows with 1,500 characters of overlap. Each (window, category) pair is one
chat-format example; every positive is kept and negatives are sampled 1:1, giving 12,412 training
examples after dropping 46 longer than 4,096 tokens.

**Prompt.** The contract excerpt comes before the category, so the 41 prompts for one window share a
prefix and vLLM's prefix cache computes the window once.

**Training.** QLoRA through Unsloth on a single T4 (fp16; the T4 has no bf16):

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen3-4B-Instruct-2507`, 4-bit |
| LoRA | r = 16, alpha = 32, dropout 0, all 7 linear projections (33M trainable, 0.81%) |
| Optimisation | lr 2e-4, 20 warm-up steps then linear decay, AdamW 8-bit, weight decay 0.01 |
| Batch | 2 × 8 gradient accumulation = 16; 776 steps (1 epoch) |
| Loss | answer tokens only |
| Time | 15.4 h over two Kaggle sessions (stop at 11 h, checkpoint, resume) |

Validation loss: 0.343 / 0.337 / 0.331 / 0.332 at steps 200 / 400 / 600 / 776. Step 600 had the lowest
loss, so both it and the final adapter were scored on the validation split; step 600 won (coverage F1
0.729 vs 0.706) and was the only model run on test.

**Evaluation.** Every model sees the same windows and prompt; per-window quotes are located in the
contract, overlapping ranges merged, and unlocatable quotes kept so they count against the model.
Baselines: the same Qwen3-4B untuned; Qwen3-8B untuned; and Qwen3-4B with hybrid retrieval (BM25 +
FAISS over `bge-small-en-v1.5` embeddings, reciprocal rank fusion, top 16 of 2,000-character passages,
one request per contract and category).

## Metrics

All computed over (contract, category) pairs; see [metrics.py](src/cuad_llm/metrics.py).

| Metric | Meaning |
| --- | --- |
| `coverage_f1` | ContractEval's headline: a true positive only if the prediction fully covers every gold span |
| `detection_f1` | Did the model correctly say the clause is present or absent? |
| `jaccard`, `token_f1` | Token overlap with the gold text, on pairs where the clause is present |
| `laziness` | Share of present clauses the model called absent ("missed clauses" above) |
| `verbatim` | Share of predicted quotes that really appear in the contract (hallucination check) |

## Limitations

- One epoch and one seed, with hyperparameters taken from a published CUAD fine-tune rather than tuned.
- Windows can't see cross-references elsewhere in a contract (e.g. a definition in section 1).
- Checkpoint selection used 40 validation contracts.
- CUAD has been public since 2021, so base models may have seen it; this applies to every model here.
- Rare categories stay weak (e.g. Price Restrictions, Source Code Escrow: 0 true positives).

## Reproduce

Local steps run on any machine; training and evals need a CUDA GPU.

```bash
uv sync                                     # Python 3.12 environment
uv run --group rag pytest                   # 24 unit tests
uv run python scripts/prepare_data.py       # download CUAD, split, build SFT data
```

**On Kaggle (free 2x T4):**

1. **Train:** import [notebooks/kaggle_train.ipynb](notebooks/kaggle_train.ipynb), run the 20-step smoke
   test, then commit the full run. It stops at 11 hours; commit again with the output attached to resume.
2. **Evaluate:** import [notebooks/kaggle_eval.ipynb](notebooks/kaggle_eval.ipynb), attach the training
   output, and choose tasks (`select_checkpoint`, `test_finetuned`, `zeroshot_4b`, `rag_4b`,
   `zeroshot_8b`). Each task is independent, and an interrupted run resumes from its response cache.

**Scoring and analysis** (after downloading each run's `predictions.jsonl` into `outputs/<run>/`):

```bash
uv run python scripts/evaluate.py outputs/finetuned_qwen3_4b outputs/zeroshot_qwen3_4b
uv run python scripts/compare_runs.py outputs/finetuned_qwen3_4b outputs/zeroshot_qwen3_4b
uv run python scripts/analyze_errors.py outputs/finetuned_qwen3_4b
```

## Layout

```
configs/          YAML configs: data, training, one per evaluated model
scripts/          prepare_data, train, predict, evaluate, compare_runs, analyze_errors, start_vllm.sh
notebooks/        Kaggle notebooks for training and evaluation
src/cuad_llm/     data, chunking, prompts, spans, metrics, inference, retrieval, pipeline
tests/            unit tests
docs/plan.md      plan, decisions and progress log
```

## Data licence

CUAD is released by The Atticus Project under CC BY 4.0.
