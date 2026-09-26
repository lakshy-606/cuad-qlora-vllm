# Plan

Fine-tune Qwen3-4B-Instruct with QLoRA (Unsloth) on CUAD and measure, rigorously, how much fine-tuning
improves clause extraction over zero-shot and retrieval baselines. Serving optimization (quantization,
latency and cost benchmarks) is out of scope. Compute: Kaggle's free GPUs (2x T4 16GB, ~30 GPU hours
a week, 12-hour sessions), so $0; 2–3 weeks part-time.

## Stack

| Component | Pick | Why |
| --- | --- | --- |
| Base model | `Qwen/Qwen3-4B-Instruct-2507` | Small models fine-tune well on extraction tasks; fast iteration |
| Zero-shot baselines | Qwen3-4B-Instruct-2507 (same model, untuned), `Qwen/Qwen3-8B` (thinking off) | Isolate the effect of fine-tuning; a larger open model for reference |
| Fine-tuning | Unsloth QLoRA | Fastest, most VRAM-efficient on one GPU |
| Eval inference | vLLM with prefix caching | A test-set eval is ~20k generations; batching plus reuse of each chunk's KV cache across the 41 category prompts keeps an eval run short |
| Tracking | Weights & Biases (free tier) | |

## Changes from the first draft of the plan

- **Eval set is the official CUAD test split (102 contracts), not ~300 held-out contracts.** This is
  the split ContractEval reports on, so our numbers compare directly with its GPT-4.1 (0.641) and
  Claude Sonnet 4 (0.523) coverage F1. Validation uses 40 contracts from the official train split.
- **Chunked inference.** Contracts run to 300k characters. Every model, baselines included, sees
  ~3k-token windows and per-window answers are merged per contract, so all configs are scored the same.
- **Prompt puts the contract excerpt before the clause category**, so the 41 prompts for one chunk
  share a prefix. vLLM's prefix cache and OpenAI's prompt caching then skip most of the prefill.
- **Serving optimization dropped.** No AWQ quantization, latency/cost benchmarks or Pareto chart; the
  project is about the fine-tune and its evaluation. vLLM stays only as the inference engine for evals.
- **Free Kaggle T4s instead of a rented A100.** T4 has no bf16, so training and inference run in fp16;
  training stops and checkpoints before the 12-hour session limit and resumes in the next session.
  Qwen3-8B is split across both T4s with tensor parallelism.
- **No paid API baseline.** A full GPT run on the test split (~57M input tokens) would cost more than
  it adds; the published GPT-4.1 and Claude Sonnet 4 ContractEval numbers are the commercial reference.
- **No hyperparameter ablations.** One fine-tune with the reference hyperparameters; the val split is
  used only to pick the best checkpoint.
- **Added a zero-shot Qwen3-4B-Instruct baseline**: the exact model before fine-tuning, so the gain is
  attributable to training rather than to model choice.
- **Qwen3-8B has no separate Instruct release**; the baseline uses the hybrid model with
  `enable_thinking: false`.

## Phases

### Phase 1: data and baselines (week 1)
- [x] Download CUAD, split, build chunk-level SFT data (`scripts/prepare_data.py`)
- [x] Metrics: coverage F1/F2, detection F1, Jaccard, token F1, laziness, verbatim, bootstrap CIs
- [x] Inference runner for any OpenAI-compatible endpoint, with resumable response cache
  (`scripts/predict.py`) and scorer (`scripts/evaluate.py`)
- [x] Baseline B retrieval: hybrid BM25 + FAISS dense retrieval with reciprocal rank fusion
- [ ] Run zero-shot Qwen3-4B-Instruct (`configs/zeroshot_qwen3_4b.yaml`), the pre-fine-tune reference
- [ ] Run baseline A: zero-shot Qwen3-8B (`configs/zeroshot_qwen3_8b.yaml`)
- [ ] Run baseline B: RAG + Qwen3-8B (`configs/rag_qwen3_8b.yaml`)

### Phase 2: fine-tune (week 2)
- [x] Training script (`scripts/train.py`): Unsloth QLoRA on Qwen3-4B-Instruct-2507, loss on assistant
  tokens only, time-limited with resume
- [x] Kaggle notebooks for training and evaluation (`notebooks/`)
- [x] Smoke test on Kaggle (20 steps): works; ~75 s/step on a T4, so 2 epochs would take ~31 h
- [x] Full training run: 1 epoch, 651 steps over two Kaggle sessions (11.0 h to step 519, then ~2.8 h);
  eval loss 0.410 → 0.396 → 0.390 → 0.389 at steps 200 / 400 / 600 / 651
- Hyperparameters: r=16, alpha=32, lr=2e-4, **1 epoch** (`configs/train.yaml`); the reference used 2,
  cut to fit the free T4 quota. W&B logging optional
- Trains on positives plus an equal number of sampled "not present" examples
- [x] Checkpoint: the final adapter (lowest val loss); the test split is used once, for the final score

### Phase 2b: retrain after a data bug
- [x] First fine-tune scored 0.585 coverage F1 vs 0.642 zero-shot, despite better detection, laziness,
  verbatim rate and span overlap. Error analysis (`scripts/analyze_errors.py`): 80% of failed clauses
  missed one of several gold spans, and the Parties category collapsed (1 of 102 fully covered).
- [x] Root cause: CUAD's train file repeats a category once per answer; the loader overwrote each
  repeat, so every train/val target kept only its last span (test data was unaffected). Fixed in
  `data.py`, verified against `CUADv1.json` (all 16,728 train pairs match), regression test added.
- [ ] Retrain on the fixed data, same hyperparameters: 12,413 examples, 776 steps, ~16.4 h on a T4
- [ ] Re-score the fine-tuned model on test

### Phase 3: evaluate and write up (week 3)
- [ ] Score the final fine-tune on test against every baseline and the published ContractEval numbers
- [ ] Reference bar: a published CUAD fine-tune reached 0.900 detection F1 vs 0.816 for zero-shot Qwen3-14B
- [ ] Per-category breakdown and error analysis: which clause types gain most, where it still fails
- [ ] README with final numbers and reproduction steps

## References

- [ContractEval (arXiv:2508.03080)](https://arxiv.org/abs/2508.03080)
- [CUAD, The Atticus Project](https://github.com/The-Atticus-Project/cuad)
- [Qwen3-4B QLoRA fine-tune on CUAD](https://github.com/Ihtesham-star/cuad_llm_finetuning) (hyperparameter starting point)
