"""QLoRA fine-tune of Qwen3-4B-Instruct on the chunk-level CUAD SFT data, with Unsloth.

Needs a CUDA GPU. Runs on a free Kaggle T4 (fp16, 16 GB) as well as bf16 GPUs. Loss is computed on the
assistant answer only. Training stops and saves a checkpoint before `max_hours`, so it survives
Kaggle's 12-hour session limit; start a new session with --resume <checkpoint dir> to continue.

Outputs (in outputs/<run_name>/):
  checkpoints/checkpoint-<step>/  LoRA checkpoints (also loadable by vLLM as adapters)
  adapter/                        final LoRA adapter + tokenizer
  train_summary.json              steps done, whether training finished, eval losses

Usage:
  python scripts/train.py --config configs/train.yaml --max-steps 20   # smoke test
  python scripts/train.py --config configs/train.yaml [--resume outputs/.../checkpoint-400]
"""

# ruff: noqa: I001  (import order matters: Unsloth must load before transformers/trl to patch them)
from __future__ import annotations

import unsloth  # noqa: F401

import argparse
import dataclasses
import inspect
import json
import os
import random
import time
from pathlib import Path

import yaml
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

from cuad_llm.data import read_jsonl

# Qwen chat-template markers; the loss is masked to everything after the assistant marker.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


class StopAfter(TrainerCallback):
    """Stop and save once `hours` have passed, leaving time to write outputs before a hard limit."""

    def __init__(self, hours: float):
        self.deadline = time.time() + hours * 3600

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() > self.deadline:
            print(f"Time budget reached at step {state.global_step}; saving and stopping.")
            control.should_training_stop = True
            control.should_save = True
        return control


def load_texts(path: Path, tokenizer, max_len: int, limit: int | None, seed: int) -> Dataset:
    rows = list(read_jsonl(path))
    if limit:
        rows = random.Random(seed).sample(rows, min(limit, len(rows)))
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    lengths = [len(ids) for ids in tokenizer(texts)["input_ids"]]
    kept = [t for t, n in zip(texts, lengths) if n <= max_len]
    print(
        f"{path.name}: {len(kept)} examples, {len(texts) - len(kept)} dropped as > {max_len} tokens"
    )
    return Dataset.from_dict({"text": kept})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--max-steps", type=int, help="stop after N steps (smoke test)")
    parser.add_argument("--resume", help="checkpoint directory to resume from")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    lora, tr = cfg["lora"], cfg["training"]
    out = Path(cfg["output_dir"]) / cfg["run_name"]
    processed = Path(cfg["processed_dir"])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
        dtype=None,  # bf16 where supported, fp16 on T4
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    train_ds = load_texts(
        processed / "sft_train.jsonl", tokenizer, cfg["max_seq_length"], None, cfg["seed"]
    )
    eval_ds = load_texts(
        processed / "sft_val.jsonl",
        tokenizer,
        cfg["max_seq_length"],
        tr["eval_examples"],
        cfg["seed"],
    )
    print("First training example as the model sees it:\n" + train_ds[0]["text"][-600:])

    # TRL renamed a few arguments across versions; pass whichever this install understands.
    config_fields = {f.name for f in dataclasses.fields(SFTConfig)}
    length_key = "max_seq_length" if "max_seq_length" in config_fields else "max_length"
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    sft_config = SFTConfig(
        output_dir=str(out / "checkpoints"),
        run_name=cfg["run_name"],
        dataset_text_field="text",
        packing=False,
        num_train_epochs=tr["epochs"],
        max_steps=args.max_steps or -1,
        learning_rate=tr["learning_rate"],
        lr_scheduler_type=tr["lr_scheduler"],
        warmup_steps=tr["warmup_steps"],
        weight_decay=tr["weight_decay"],
        optim="adamw_8bit",
        per_device_train_batch_size=tr["per_device_batch_size"],
        per_device_eval_batch_size=tr["per_device_batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=tr["logging_steps"],
        eval_strategy="steps",
        eval_steps=tr["save_steps"],
        save_strategy="steps",
        save_steps=tr["save_steps"],
        save_total_limit=tr["save_total_limit"],
        report_to="wandb" if use_wandb else "none",
        seed=cfg["seed"],
        **{length_key: cfg["max_seq_length"]},
    )
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    tokenizer_key = "processing_class" if "processing_class" in trainer_params else "tokenizer"
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[StopAfter(tr["max_hours"])],
        **{tokenizer_key: tokenizer},
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART
    )

    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=args.resume)
    hours = (time.time() - t0) / 3600
    samples_per_s = result.metrics.get("train_samples_per_second", 0.0)

    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    state = trainer.state
    summary = {
        "global_step": state.global_step,
        "max_steps": state.max_steps,
        "finished": state.global_step >= state.max_steps,
        "hours_this_session": round(hours, 2),
        "train_examples": len(train_ds),
        "train_samples_per_second": samples_per_s,
        # From this session's speed; a smoke test uses it to size the full run against time limits.
        "est_hours_full_run": round(len(train_ds) * tr["epochs"] / samples_per_s / 3600, 1)
        if samples_per_s
        else None,
        "eval_losses": [
            {"step": h["step"], "eval_loss": h["eval_loss"]}
            for h in state.log_history
            if "eval_loss" in h
        ],
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not summary["finished"]:
        print(f"Not finished: resume with --resume {out / 'checkpoints'}/checkpoint-<last step>")


if __name__ == "__main__":
    main()
